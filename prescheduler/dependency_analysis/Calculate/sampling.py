import logging
import numpy as np
from typing import Dict, Optional, Union, List

from Calculate.inference import log_likelihood
from Learning.validity import is_valid
from Structure.nodes import Independent, Decomposition, Joint, get_nodes_by_type, eval_spn_top_down

logger = logging.getLogger(__name__)


def merge_input_vals(input_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """合并输入值列表"""
    if not input_list or any(x is None for x in input_list):
        return None
    return np.concatenate(input_list)


def sample_prod(node, input_vals: List[np.ndarray], data: Optional[np.ndarray] = None, 
               lls_per_node: Optional[np.ndarray] = None, 
               rand_gen: Optional[np.random.Generator] = None) -> Optional[Dict]:
    """乘积节点采样"""
    merged_vals = merge_input_vals(input_vals)
    if merged_vals is None:
        return None

    # 所有子节点使用相同的输入
    return {c: merged_vals for c in node.children}


def sample_sum(node, input_vals: List[np.ndarray], data: Optional[np.ndarray] = None, 
              lls_per_node: Optional[np.ndarray] = None, 
              rand_gen: Optional[np.random.Generator] = None) -> Optional[Dict]:
    """求和节点采样"""
    merged_vals = merge_input_vals(input_vals)
    if merged_vals is None:
        return None

    num_samples = len(merged_vals)
    num_children = len(node.children)
    
    # 计算加权对数概率
    w_children_log_probs = np.zeros((num_samples, num_children))
    log_weights = np.log(node.weights)
    
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = lls_per_node[merged_vals, c.id] + log_weights[i]

    # Gumbel-Max采样
    gumbels = rand_gen.gumbel(size=(num_samples, num_children))
    g_children_log_probs = w_children_log_probs + gumbels
    rand_child_branches = np.argmax(g_children_log_probs, axis=1)

    # 分配样本到子节点
    children_row_ids = {}
    for i, c in enumerate(node.children):
        mask = (rand_child_branches == i)
        children_row_ids[c] = merged_vals[mask] if np.any(mask) else np.array([], dtype=int)

    return children_row_ids


def sample_leaf(node, input_vals: List[np.ndarray], data: Optional[np.ndarray] = None, 
               lls_per_node: Optional[np.ndarray] = None, 
               rand_gen: Optional[np.random.Generator] = None) -> None:
    """叶节点采样"""
    merged_vals = merge_input_vals(input_vals)
    if merged_vals is None:
        return

    # 找到需要采样的NaN位置
    data_subset = data[merged_vals, node.scope]
    nan_mask = np.isnan(data_subset).flatten()
    
    if not np.any(nan_mask):
        return

    # 获取采样函数并执行采样
    sample_func = _leaf_sampling.get(type(node))
    if sample_func is None:
        logger.warning(f"No sampling function found for node type {type(node)}")
        return
    
    nan_indices = merged_vals[nan_mask]
    samples = sample_func(node, n_samples=len(nan_indices), 
                         data=data[nan_indices, :], rand_gen=rand_gen)
    
    data[nan_indices, node.scope] = samples


_node_sampling = {Independent: sample_prod, Joint: sample_sum}
_leaf_sampling = {}


def add_leaf_sampling(node_type, lambda_func):
    _leaf_sampling[node_type] = lambda_func
    _node_sampling[node_type] = sample_leaf


def add_node_sampling(node_type, lambda_func):
    _node_sampling[node_type] = lambda_func


def sample_instances(node, input_data: np.ndarray, rand_gen: np.random.Generator, 
                    node_sampling: Dict = None, in_place: bool = False) -> np.ndarray:
    """
    分层采样实现
    
    Args:
        node: SPN根节点
        input_data: 输入数据，包含NaN值表示需要采样的位置
        rand_gen: 随机数生成器
        node_sampling: 节点采样函数映射
        in_place: 是否就地修改输入数据
    
    Returns:
        采样后的数据
    """
    if node_sampling is None:
        node_sampling = _node_sampling
    
    # 验证SPN结构
    valid, err = is_valid(node)
    if not valid:
        raise ValueError(f"Invalid SPN: {err}")

    # 验证输入数据
    if not np.all(np.any(np.isnan(input_data), axis=1)):
        raise ValueError("Each row must have at least one NaN value for sampling")

    # 准备数据
    data = input_data if in_place else np.copy(input_data)
    
    # 获取所有节点并初始化似然矩阵
    nodes = get_nodes_by_type(node)
    lls_per_node = np.zeros((data.shape[0], len(nodes)), dtype=data.dtype)

    # 自底向上计算似然
    log_likelihood(node, data, dtype=data.dtype, lls_matrix=lls_per_node)

    # 自顶向下采样
    instance_ids = np.arange(data.shape[0])
    eval_spn_top_down(node, node_sampling, parent_result=instance_ids, 
                     data=data, lls_per_node=lls_per_node, rand_gen=rand_gen)

    return data