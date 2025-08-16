import logging
import numpy as np
from scipy.special import logsumexp
from typing import Dict, Any, Optional, Callable, Union

from Structure.nodes import Independent, Joint, Decomposition, eval_spn_bottom_up

logger = logging.getLogger(__name__)

# 统一常量定义
EPSILON = np.finfo(np.float64).eps
MIN_LOG_VALUE = np.finfo(np.float64).min


def leaf_marginalized_likelihood(node, data: np.ndarray = None, dtype: np.dtype = np.float64, 
                                log_space: bool = False, **kwargs) -> tuple:
    """计算叶节点的边际化似然"""
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    if log_space:
        probs.fill(0.0)
    
    data_subset = data[:, node.scope]
    marg_ids = np.isnan(data_subset)
    observations = data_subset[~marg_ids]
    
    return probs, marg_ids, observations


def prod_log_likelihood(node, children: list, dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
    """计算乘积节点的对数似然"""
    llchildren = np.column_stack(children)
    pll = np.sum(llchildren, axis=1, dtype=dtype)
    # 使用更安全的无穷值处理
    pll = np.clip(pll, MIN_LOG_VALUE, None)
    return pll


def prod_likelihood(node, children: list, dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
    """计算乘积节点的似然"""
    llchildren = np.column_stack(children)
    return np.prod(llchildren, axis=1, dtype=dtype)


def sum_log_likelihood(node, children: list, dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
    """计算求和节点的对数似然"""
    llchildren = np.column_stack(children)
    
    # 验证权重归一化
    weights_sum = np.sum(node.weights)
    if not np.isclose(weights_sum, 1.0, rtol=1e-10):
        logger.warning(f"Unnormalized weights {node.weights} for node {node}")
    
    weights = np.asarray(node.weights, dtype=dtype)
    return logsumexp(llchildren, b=weights, axis=1)


def sum_likelihood(node, children: list, dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
    """计算求和节点的似然"""
    llchildren = np.column_stack(children)
    
    weights_sum = np.sum(node.weights)
    if not np.isclose(weights_sum, 1.0, rtol=1e-10):
        logger.warning(f"Unnormalized weights {node.weights} for node {node}")
    
    weights = np.asarray(node.weights, dtype=dtype)
    return np.dot(llchildren, weights)


def factorize_likelihood(node, r_children: list, l_children: list, 
                        dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
    """计算分解节点的似然"""
    if len(r_children) != len(l_children):
        raise ValueError("probability shape mismatch")
    
    r_children = np.column_stack(r_children)
    l_children = np.column_stack(l_children)
    
    return (r_children * l_children).flatten()


def factorize_log_likelihood(node, r_children: list, l_children: list,
                            dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
    """计算分解节点的对数似然"""
    if len(r_children) != len(l_children):
        raise ValueError("probability shape mismatch")
    
    r_children = np.column_stack(r_children)
    l_children = np.column_stack(l_children)
    
    pll = r_children + l_children
    return np.clip(pll.flatten(), MIN_LOG_VALUE, None)


_node_log_likelihood = {Joint: sum_log_likelihood, Independent: prod_log_likelihood, Decomposition: factorize_log_likelihood}
_node_likelihood = {Joint: sum_likelihood, Independent: prod_likelihood, Decomposition: factorize_likelihood}


def _get_exp_likelihood(f_log):
    def f_exp(node, *args, **kwargs):
        return np.exp(f_log(node, *args, **kwargs))

    return f_exp


def _get_log_likelihood(f_exp):
    def f_log(node, *args, **kwargs):
        with np.errstate(divide="ignore"):
            nll = np.log(f_exp(node, *args, **kwargs))
            nll = np.clip(nll, MIN_LOG_VALUE, None)
            if np.any(np.isnan(nll)):
                raise ValueError("Log likelihood contains NaN values")
            return nll

    return f_log


def add_node_likelihood(node_type: type, lambda_func: Optional[Callable] = None, 
                       log_lambda_func: Optional[Callable] = None):
    """添加节点似然计算函数"""
    if lambda_func is None and log_lambda_func is None:
        raise ValueError("At least one of lambda_func or log_lambda_func must be provided")

    if lambda_func is None:
        lambda_func = _get_exp_likelihood(log_lambda_func)
    _node_likelihood[node_type] = lambda_func

    if log_lambda_func is None:
        log_lambda_func = _get_log_likelihood(lambda_func)
    _node_log_likelihood[node_type] = log_lambda_func


def likelihood(node, data: np.ndarray, dtype: np.dtype = np.float64, 
              node_likelihood: Dict = None, lls_matrix: Optional[np.ndarray] = None, 
              debug: bool = False, **kwargs) -> np.ndarray:
    """计算SPN的似然"""
    if node_likelihood is None:
        node_likelihood = _node_likelihood
        
    # 输入验证
    if len(data.shape) != 2:
        raise ValueError(f"data must be 2D, found: {data.shape}")
    
    all_results = {}

    if debug:
        original_node_likelihood = node_likelihood
        def exec_funct(node, *args, **kwargs):
            if node is None:
                raise ValueError("node is None")
            funct = original_node_likelihood[type(node)]
            ll = funct(node, *args, **kwargs)
            expected_shape = (data.shape[0],)
            if ll.shape != expected_shape:
                raise ValueError(f"node {node.id} result shape {ll.shape} != expected {expected_shape}")
            if np.any(np.isnan(ll)):
                raise ValueError(f"ll contains NaN for node {node.id}")
            return ll

        node_likelihood = {k: exec_funct for k in node_likelihood.keys()}

    result = eval_spn_bottom_up(node, node_likelihood, all_results=all_results, 
                               debug=debug, dtype=dtype, data=data, **kwargs)

    # 更高效的结果存储
    if lls_matrix is not None:
        for n, ll in all_results.items():
            lls_matrix[:, n.id] = ll.flatten() if ll.ndim > 1 else ll

    return result


def log_likelihood(node, data: np.ndarray, dtype: np.dtype = np.float64, 
                  node_log_likelihood: Dict = None, lls_matrix: Optional[np.ndarray] = None, 
                  debug: bool = False, **kwargs) -> np.ndarray:
    """计算SPN的对数似然"""
    if node_log_likelihood is None:
        node_log_likelihood = _node_log_likelihood
    return likelihood(node, data, dtype=dtype, node_likelihood=node_log_likelihood, 
                     lls_matrix=lls_matrix, debug=debug, **kwargs)


def conditional_log_likelihood(node_joint, node_marginal, data: np.ndarray, 
                              log_space: bool = True, dtype: np.dtype = np.float64) -> np.ndarray:
    """计算条件对数似然"""
    joint_ll = log_likelihood(node_joint, data, dtype)
    marginal_ll = log_likelihood(node_marginal, data, dtype)
    result = joint_ll - marginal_ll
    
    return result if log_space else np.exp(result)


