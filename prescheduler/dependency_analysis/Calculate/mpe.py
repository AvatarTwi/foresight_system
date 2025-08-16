import numpy as np
import logging
from typing import Dict, Optional, List, Union

from Calculate.inference import log_likelihood, sum_log_likelihood, prod_log_likelihood
from Learning.validity import is_valid
from Structure.nodes import Independent, Joint, Decomposition, get_nodes_by_type, eval_spn_top_down

logger = logging.getLogger(__name__)


def merge_input_vals(input_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """合并输入值列表"""
    if not input_list or any(x is None for x in input_list):
        return None
    return np.concatenate(input_list)


def mpe_prod(node, parent_result: List[np.ndarray], data: Optional[np.ndarray] = None, 
            lls_per_node: Optional[np.ndarray] = None, rand_gen=None) -> Optional[Dict]:
    """乘积节点MPE推理"""
    merged_result = merge_input_vals(parent_result)
    if merged_result is None:
        return None

    # 乘积节点：所有子节点都选择
    return {c: merged_result for c in node.children}


def mpe_sum(node, parent_result: List[np.ndarray], data: Optional[np.ndarray] = None, 
           lls_per_node: Optional[np.ndarray] = None, rand_gen=None) -> Optional[Dict]:
    """求和节点MPE推理"""
    merged_result = merge_input_vals(parent_result)
    if merged_result is None:
        return None

    num_samples = len(merged_result)
    num_children = len(node.children)
    
    # 计算加权对数概率
    w_children_log_probs = np.zeros((num_samples, num_children))
    log_weights = np.log(np.asarray(node.weights))
    
    for i, c in enumerate(node.children):
        child_lls = lls_per_node[merged_result, c.id]
        w_children_log_probs[:, i] = child_lls + log_weights[i]

    # 选择最大概率的子节点
    max_child_branches = np.argmax(w_children_log_probs, axis=1)

    # 分配样本到对应的最优子节点
    children_row_ids = {}
    for i, c in enumerate(node.children):
        mask = (max_child_branches == i)
        children_row_ids[c] = merged_result[mask] if np.any(mask) else np.array([], dtype=int)

    return children_row_ids


def get_mpe_top_down_leaf(node, input_vals: List[np.ndarray], data: Optional[np.ndarray] = None, 
                         mode: float = 0.0) -> None:
    """叶节点MPE推理"""
    merged_vals = merge_input_vals(input_vals)
    if merged_vals is None:
        return

    # 找到需要填充的NaN位置
    data_subset = data[merged_vals, node.scope]
    nan_mask = np.isnan(data_subset).flatten()
    
    if not np.any(nan_mask):
        return

    # 用模式值填充NaN
    nan_indices = merged_vals[nan_mask]
    data[nan_indices, node.scope] = mode


_node_top_down_mpe = {Independent: mpe_prod, Joint: mpe_sum}
_node_bottom_up_mpe_log = {Joint: sum_log_likelihood, Independent: prod_log_likelihood}


def add_node_mpe(node_type, log_bottom_up_lambda, top_down_lambda):
    _node_top_down_mpe[node_type] = top_down_lambda
    _node_bottom_up_mpe_log[node_type] = log_bottom_up_lambda


def mpe(node, input_data: np.ndarray, 
        node_top_down_mpe: Dict = None,
        node_bottom_up_mpe_log: Dict = None,
        in_place: bool = False) -> np.ndarray:
    """
    最大后验推理(MPE)
    
    Args:
        node: SPN根节点
        input_data: 输入数据，包含NaN值表示需要推理的位置
        node_top_down_mpe: 自顶向下MPE函数映射
        node_bottom_up_mpe_log: 自底向上对数似然函数映射
        in_place: 是否就地修改输入数据
    
    Returns:
        MPE推理后的数据
    """
    if node_top_down_mpe is None:
        node_top_down_mpe = _node_top_down_mpe
    if node_bottom_up_mpe_log is None:
        node_bottom_up_mpe_log = _node_bottom_up_mpe_log
    
    # 验证SPN结构
    valid, err = is_valid(node)
    if not valid:
        raise ValueError(f"Invalid SPN: {err}")

    # 验证输入数据
    if not np.all(np.any(np.isnan(input_data), axis=1)):
        raise ValueError("Each row must have at least one NaN value for MPE inference")

    # 准备数据
    data = input_data if in_place else np.copy(input_data)
    
    # 获取所有节点并初始化似然矩阵
    nodes = get_nodes_by_type(node)
    lls_per_node = np.zeros((data.shape[0], len(nodes)), dtype=data.dtype)

    # 自底向上计算对数似然
    log_likelihood(node, data, dtype=data.dtype, 
                  node_log_likelihood=node_bottom_up_mpe_log, 
                  lls_matrix=lls_per_node)

    # 自顶向下进行MPE推理
    instance_ids = np.arange(data.shape[0])
    eval_spn_top_down(node, node_top_down_mpe, parent_result=instance_ids, 
                     data=data, lls_per_node=lls_per_node)

    return data