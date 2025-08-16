import time
import copy
import numpy as np
from typing import Optional, Tuple, List, Union
from Learning.utils import convert_to_scope_domain, validate_data_consistency
from sklearn.cluster import KMeans
from Learning.splitting.RDC import rdc_test
from Structure.nodes import Independent, Joint, Decomposition, Leaf
from Structure.leaves.aspn_leaves.Multi_Histograms import Multi_histogram, multidim_cumsum
from Structure.leaves.aspn_leaves.Histograms import Histogram
from Structure.leaves.aspn_leaves.Merge_leaves import Merge_leaves
from Calculate.inference import EPSILON

# 常量定义
DEFAULT_RDC_SAMPLE_SIZE = 50000
DEFAULT_RDC_K_VALUE = 10

def calculate_RDC(data: np.ndarray, ds_context, scope: List[int], condition: List[int], 
                  sample_size: int = DEFAULT_RDC_SAMPLE_SIZE) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    使用数据计算RDC邻接矩阵（优化版本）
    """
    start_time = time.time()
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    meta_types = ds_context.get_meta_types_by_scope(scope_range)
    domains = ds_context.get_domains_by_scope(scope_range)

    # 智能采样策略
    if len(data) <= sample_size:
        sample_data = data
    else:
        # 使用更高效的随机采样
        sample_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
        sample_data = data[sample_indices]
    
    # 计算RDC并处理NaN
    rdc_adjacency_matrix = rdc_test(sample_data, meta_types, domains, k=DEFAULT_RDC_K_VALUE)
    rdc_adjacency_matrix = np.nan_to_num(rdc_adjacency_matrix, nan=0.0)
    
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # 只有耗时较长时才记录
        logging.debug(f"RDC计算耗时: {elapsed_time:.3f}秒, 样本数: {len(sample_data)}")
    
    return rdc_adjacency_matrix, scope_loc, condition_loc


def _validate_aspn_data_shape(aspn, data: np.ndarray, operation_name: str) -> None:
    """优化的ASPN数据形状验证"""
    if data is None:
        return
        
    if hasattr(aspn, 'range') and aspn.range:
        expected_cols = len(aspn.scope) + len(aspn.range)
    else:
        expected_cols = len(aspn.scope)
    
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"数据形状不匹配 ({operation_name}): "
            f"期望{expected_cols}列, 实际{data.shape[1]}列"
        )


def top_down_update(aspn, ds_context, data_insert: Optional[np.ndarray] = None, 
                   data_delete: Optional[np.ndarray] = None) -> None:
    """
    优化后的自顶向下更新函数
    """
    # 统一数据验证
    for data, op_name in [(data_insert, "insert"), (data_delete, "delete")]:
        if data is not None:
            _validate_aspn_data_shape(aspn, data, op_name)

    # 叶节点处理
    if isinstance(aspn, Leaf):
        update_leaf(aspn, ds_context, data_insert, data_delete)
        return

    # 分解节点处理
    if isinstance(aspn, Decomposition):
        _handle_decomposition_update(aspn, ds_context, data_insert, data_delete)
        return

    # 分割节点处理
    if isinstance(aspn, Joint) and hasattr(aspn, 'range') and aspn.range is not None:
        _handle_split_node_update(aspn, ds_context, data_insert, data_delete)
        return

    # 求和节点处理
    if isinstance(aspn, Joint):
        _handle_sum_node_update(aspn, ds_context, data_insert, data_delete)
        return

    # 独立乘积节点处理
    if isinstance(aspn, Independent):
        _handle_product_node_update(aspn, ds_context, data_insert, data_delete)


def _handle_decomposition_update(aspn, ds_context, data_insert, data_delete):
    """处理分解节点的更新"""
    left_cols = [aspn.scope.index(i) for i in aspn.children[0].scope]
    left_insert = data_insert[:, left_cols] if data_insert is not None else None
    left_delete = data_delete[:, left_cols] if data_delete is not None else None
    
    top_down_update(aspn.children[0], ds_context, left_insert, left_delete)
    top_down_update(aspn.children[1], ds_context, data_insert, data_delete)


def _handle_split_node_update(aspn, ds_context, data_insert, data_delete):
    """处理分割节点的更新"""
    if aspn.cluster_centers:
        raise ValueError("分割节点不应有聚类中心")
        
    for child in aspn.children:
        if not hasattr(child, 'range') or child.range is None:
            raise ValueError("分割节点的子节点必须有range属性")
            
        new_data_insert = split_data_by_range(data_insert, child.range, child.scope) if data_insert else None
        new_data_delete = split_data_by_range(data_delete, child.range, child.scope) if data_delete else None
        top_down_update(child, ds_context, new_data_insert, new_data_delete)


def _handle_sum_node_update(aspn, ds_context, data_insert, data_delete):
    """处理求和节点的更新"""
    if not aspn.cluster_centers:
        raise ValueError("求和节点必须有聚类中心")
        
    # 记录原始基数
    origin_cardinality = getattr(aspn, 'cardinality', 0)
    
    # 更新基数
    insert_len = len(data_insert) if data_insert is not None else 0
    delete_len = len(data_delete) if data_delete is not None else 0
    aspn.cardinality = max(0, origin_cardinality + insert_len - delete_len)

    # 分割数据
    num_children = len(aspn.children)
    new_data_insert = (split_data_by_cluster_center(data_insert, aspn.cluster_centers) 
                      if data_insert is not None else [None] * num_children)
    new_data_delete = (split_data_by_cluster_center(data_delete, aspn.cluster_centers) 
                      if data_delete is not None else [None] * num_children)

    # 更新权重和递归处理子节点
    for i, child in enumerate(aspn.children):
        dl_insert = len(new_data_insert[i]) if new_data_insert[i] is not None else 0
        dl_delete = len(new_data_delete[i]) if new_data_delete[i] is not None else 0
        
        child_cardinality = max(0, origin_cardinality * aspn.weights[i] + dl_insert - dl_delete)
        aspn.weights[i] = child_cardinality / aspn.cardinality if aspn.cardinality > 0 else 0
        
        top_down_update(child, ds_context, new_data_insert[i], new_data_delete[i])


def _handle_product_node_update(aspn, ds_context, data_insert, data_delete):
    """处理独立乘积节点的更新"""
    for child in aspn.children:
        index = [aspn.scope.index(s) for s in child.scope]
        new_data_insert = data_insert[:, index] if data_insert is not None else None
        new_data_delete = data_delete[:, index] if data_delete is not None else None
        top_down_update(child, ds_context, new_data_insert, new_data_delete)


def _extract_dataset_for_leaf(aspn, dataset: np.ndarray) -> np.ndarray:
    """优化的叶节点数据提取函数"""
    if not hasattr(aspn, 'range') or not aspn.range:
        return dataset
    
    total_attrs = aspn.scope + aspn.condition
    range_attrs = list(aspn.range.keys())
    
    # 确定数据列对应关系
    if dataset.shape[1] == len(total_attrs):
        idx = sorted(total_attrs)
        keep_indices = [idx.index(i) for i in aspn.scope]
    elif dataset.shape[1] == len(aspn.scope + range_attrs):
        idx = sorted(aspn.scope + range_attrs)
        keep_indices = [idx.index(i) for i in aspn.scope]
    else:
        raise ValueError(
            f"数据维度不匹配: 数据有{dataset.shape[1]}列, "
            f"但期望{len(total_attrs)}或{len(aspn.scope + range_attrs)}列"
        )
    
    return dataset[:, keep_indices]


def _safe_update_nan_percentage(aspn, old_card: int, new_card: int, new_card_actual: int) -> float:
    """安全的NaN百分比更新函数"""
    if new_card == 0:
        return 0.0
        
    new_nan_perc = new_card_actual / new_card
    total_card = new_card + old_card
    
    if total_card == 0:
        aspn.nan_perc = 0.0
        return 0.0
    
    old_weight = old_card / total_card
    new_weight = new_card / total_card
    
    aspn.nan_perc = old_weight * getattr(aspn, 'nan_perc', 0.0) + new_weight * new_nan_perc
    
    total_actual = new_card_actual + old_card * aspn.nan_perc
    return new_card_actual / total_actual if total_actual > 0 else 0.0


def insert_leaf_Histogram(aspn, ds_context, dataset: np.ndarray) -> None:
    """优化后的直方图叶节点插入函数"""
    dataset = _extract_dataset_for_leaf(aspn, dataset)
    
    if len(dataset) == 0:
        return
    
    # 处理NaN值
    valid_mask = ~np.isnan(dataset.flatten())
    valid_data = dataset.flatten()[valid_mask]
    
    new_card = len(dataset)
    new_card_actual = len(valid_data)
    
    # 更新基数和NaN百分比
    old_card = getattr(aspn, 'cardinality', 0)
    aspn.cardinality = old_card + new_card
    new_weight = _safe_update_nan_percentage(aspn, old_card, new_card, new_card_actual)
    
    if new_card_actual == 0:
        return
    
    old_weight = 1 - new_weight
    
    # 优化分割点扩展
    current_breaks = list(aspn.breaks)
    data_min, data_max = np.min(valid_data), np.max(valid_data)
    
    # 检查是否需要扩展边界
    left_expansion = data_min < current_breaks[0]
    right_expansion = data_max > current_breaks[-1]
    
    if left_expansion:
        current_breaks.insert(0, data_min - EPSILON)
    if right_expansion:
        current_breaks.append(data_max + EPSILON)

    # 计算直方图
    new_pdf, _ = np.histogram(valid_data, bins=current_breaks)
    new_pdf = new_pdf.astype(np.float64)
    
    # 归一化
    pdf_sum = np.sum(new_pdf)
    if pdf_sum > 0:
        new_pdf /= pdf_sum
    
    # 调整旧PDF以匹配新的分割点
    old_pdf = np.array(aspn.pdf, dtype=np.float64)
    if left_expansion:
        old_pdf = np.insert(old_pdf, 0, 0.0)
    if right_expansion:
        old_pdf = np.append(old_pdf, 0.0)

    # 合并PDF
    final_pdf = old_weight * old_pdf + new_weight * new_pdf
    final_cdf = np.concatenate([[0], np.cumsum(final_pdf)])
    
    # 验证并更新
    pdf_sum = np.sum(final_pdf)
    if not np.isclose(pdf_sum, 1.0, rtol=1e-10):
        final_pdf /= pdf_sum  # 重新归一化
        final_cdf = np.concatenate([[0], np.cumsum(final_pdf)])
    
    aspn.breaks = np.array(current_breaks)
    aspn.pdf = final_pdf
    aspn.cdf = final_cdf


def split_data_by_cluster_center(dataset: Optional[np.ndarray], centers: List, seed: int = 17) -> List[np.ndarray]:
    """
    优化的基于聚类中心的数据分割函数
    """
    if dataset is None or len(dataset) == 0:
        return [np.array([]).reshape(0, dataset.shape[1] if dataset is not None else 0)] * len(centers)
    
    k = len(centers)
    if k == 0:
        return []
    
    # 使用预训练的聚类中心
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=1)
    kmeans.cluster_centers_ = np.asarray(centers)
    
    try:
        cluster_labels = kmeans.predict(dataset)
    except Exception as e:
        logging.warning(f"聚类预测失败: {e}")
        # 回退到简单分割
        return [dataset] if k == 1 else [dataset[::k] for _ in range(k)]
    
    # 按聚类标签分组数据
    result = []
    unique_labels = np.sort(np.unique(cluster_labels))
    
    for label in range(k):
        if label in unique_labels:
            mask = cluster_labels == label
            result.append(dataset[mask])
        else:
            # 处理空聚类的情况
            result.append(np.array([]).reshape(0, dataset.shape[1]))
    
    return result
