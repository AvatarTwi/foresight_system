import logging
import copy
import multiprocessing
import os
import time
from collections import deque
from enum import Enum
from typing import Optional, List, Callable
from contextlib import contextmanager

from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_array
from Learning.utils import convert_to_scope_domain, get_matached_domain, validate_data_consistency
from Learning.statistics import get_structure_stats

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

from Learning.transformStructure import Prune
from Learning.validity import is_valid
from Learning.splitting.RDC import rdc_test
from Structure.nodes import Independent, Joint, Decomposition, assign_ids

parallel = True

@contextmanager
def managed_pool():
    """进程池上下文管理器"""
    pool = None
    try:
        if parallel:
            cpus = max(1, os.cpu_count() - 6)
        else:
            cpus = 1
        pool = multiprocessing.Pool(processes=cpus)
        yield pool
    finally:
        if pool:
            pool.close()
            pool.join()


def calculate_RDC(data: np.ndarray, ds_context, scope: List[int], condition: List[int], 
                  sample_size: int) -> tuple:
    """
    优化的RDC计算函数
    """
    start_time = time.time()
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    meta_types = ds_context.get_meta_types_by_scope(scope_range)
    domains = ds_context.get_domains_by_scope(scope_range)

    # 智能采样
    sample_data = data if len(data) <= sample_size else data[
        np.random.randint(data.shape[0], size=sample_size)
    ]
    
    rdc_adjacency_matrix = rdc_test(sample_data, meta_types, domains, k=10)
    rdc_adjacency_matrix = np.nan_to_num(rdc_adjacency_matrix, nan=0.0)
    
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        logging.debug(f"RDC计算耗时: {elapsed:.3f}秒, 样本数: {len(sample_data)}")
    
    return rdc_adjacency_matrix, scope_loc, condition_loc


class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4  # This refers to consider the variables in the scope as independent
    REMOVE_UNINFORMATIVE_FEATURES = 5  # If all data of certain attribute are the same
    FACTORIZE = 6  # A factorized node
    REMOVE_CONDITION = 7  # Remove independent set from the condition
    SPLIT_ROWS_CONDITION = 8
    SPLIT_COLUMNS_CONDITION = 9  # NOT IMPLEMENTED Split rows when there is condition, using conditional independence
    FACTORIZE_CONDITION = 10  # NOT IMPLEMENTED Decomposition columns when there is condition, using conditional independence


def get_next_operation(ds_context, min_instances_slice=100, min_features_slice=1, multivariate_leaf=True,
                       threshold=0.3, rdc_sample_size=50000, rdc_strong_connection_threshold=0.75):
    """
    :param ds_context: A context specifying the type of the variables in the dataset
    :param min_instances_slice: minimum number of rows to stop splitting by rows
    :param min_features_slice: minimum number of feature to stop splitting by columns usually 1
    :param multivariate_leaf: If true, we fit joint distribution with multivariates.
                              This only controls the SPN branch.
    :return: return a function, call to which will generate the next operation to perform
    """

    def next_operation(
            data,
            scope,
            condition,
            no_clusters=False,
            no_independencies=False,
            no_condition=False,
            is_strong_connected=False,
            rdc_threshold=threshold,
            rdc_strong_connection_threshold=rdc_strong_connection_threshold
    ):

        """
        :param data: local data set
        :param scope: scope of parent node
        :param condition: scope of conditional of parent node
        :param no_clusters: if True, we are assuming the highly correlated and unseparable data
        :param no_independencies: if True, cannot split by columns
        :param no_condition: if True, cannot remove conditions
        :param rdc_threshold: Under which, we will consider the two variables as independent
        :param rdc_strong_connection_threshold: Above which, we will consider the two variables as strongly related
        :return: Next Operation and parameters to call if have any
        """

        assert len(set(scope).intersection(set(condition))) == 0, "scope and condition mismatch"
        assert (len(scope) + len(condition)) == data.shape[1], "Redundant data columns"

        minimalFeatures = len(scope) == min_features_slice
        minimalInstances = data.shape[0] <= min_instances_slice

        if minimalFeatures and len(condition) == 0:
            return Operation.CREATE_LEAF, None

        if is_strong_connected and (len(condition) == 0):
            # the case of strongly connected components, directly model them
            return Operation.CREATE_LEAF, None

        if (minimalInstances and len(condition) == 0) or (no_clusters and len(condition) <= 1):
            if multivariate_leaf or is_strong_connected:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        # Check if all data of an attribute has the same value (very possible for categorical data)
        uninformative_features_idx = np.var(data, 0) == 0
        ncols_zero_variance = np.sum(uninformative_features_idx)
        if ncols_zero_variance > 0:
            if ncols_zero_variance == data.shape[1]:
                if multivariate_leaf:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            else:
                feature_idx = np.asarray(sorted(scope + condition))
                uninformative_features = list(feature_idx[uninformative_features_idx])
                if set(uninformative_features) == set(scope):
                    if multivariate_leaf:
                        return Operation.CREATE_LEAF, None
                    else:
                        return Operation.NAIVE_FACTORIZATION, None
                if len(condition) == 0 or len(set(uninformative_features).intersection(set(condition))) != 0:
                    # This is very messy here but essentially realigning the scope and condition with the data column
                    return (
                        Operation.REMOVE_UNINFORMATIVE_FEATURES,
                        (get_matached_domain(uninformative_features_idx, scope, condition))
                    )

        if len(condition) != 0 and no_condition:
            """
                In this case, we have no condition to remove. Must split rows or create leaf.
            """
            if minimalInstances:
                if multivariate_leaf or is_strong_connected:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            elif not no_clusters:
                return Operation.SPLIT_ROWS_CONDITION, calculate_RDC(data, ds_context, scope, condition,
                                                                     rdc_sample_size)

        elif len(condition) != 0:
            """Try to eliminate some of condition, which are independent of scope
            """
            rdc_adjacency_matrix, scope_loc, condition_loc = calculate_RDC(data, ds_context, scope, condition,
                                                                           rdc_sample_size)
            independent_condition = []
            remove_cols = []
            for i in range(len(condition_loc)):
                cond = condition_loc[i]
                is_indep = True
                for s in scope_loc:
                    if rdc_adjacency_matrix[cond][s] > rdc_threshold:
                        is_indep = False
                        continue
                if is_indep:
                    remove_cols.append(cond)
                    independent_condition.append(condition[i])

            if len(independent_condition) != 0:
                return Operation.REMOVE_CONDITION, (independent_condition, remove_cols)

            else:
                # If there is nothing to eliminate from conditional set, we split rows
                if minimalInstances:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.SPLIT_ROWS_CONDITION, (rdc_adjacency_matrix, scope_loc, condition_loc)


        elif not no_clusters and not minimalInstances:
            """In this case:  len(condition) == 0 and not minimalFeatures and not no_clusters
               So we try to split rows or factorize
            """
            rdc_adjacency_matrix, scope_loc, _ = calculate_RDC(data, ds_context, scope, condition, rdc_sample_size)
            if not no_independencies:
                # test independence
                rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0

                num_connected_comp = 0
                indep_res = np.zeros(data.shape[1])
                for i, c in enumerate(connected_components(from_numpy_array(rdc_adjacency_matrix))):
                    indep_res[list(c)] = i + 1
                    num_connected_comp += 1
                if num_connected_comp > 1:
                    # there exists independent sets, split by columns
                    return Operation.SPLIT_COLUMNS, indep_res

            rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_strong_connection_threshold] = 0
            strong_connected_comp = []  # strongly connected components
            for c in connected_components(from_numpy_array(rdc_adjacency_matrix)):
                if len(c) > 1:
                    component = list(c)
                    component.sort()
                    for i in range(len(c)):
                        component[i] = scope[component[i]]
                    strong_connected_comp.append(component)

            if len(strong_connected_comp) != 0:
                if strong_connected_comp[0] == scope:
                    # the whole scope is actually strongly connected
                    return Operation.CREATE_LEAF, None
                # there exists sets of strongly connect component, must factorize them out
                return Operation.FACTORIZE, strong_connected_comp

        elif minimalInstances:
            if multivariate_leaf or is_strong_connected:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        # if none of the above conditions follows, we split by row and try again.
        if len(condition) == 0:
            return Operation.SPLIT_ROWS, None
        else:
            return Operation.SPLIT_ROWS_CONDITION, calculate_RDC(data, ds_context, scope, condition, rdc_sample_size)

    return next_operation


def default_slicer(data, cols, num_cond_cols=None):
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))

        return data[:, cols]
    else:
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)


def learn_structure_binary(dataset: np.ndarray, ds_context, split_rows: Callable, split_rows_condition: Callable,
                          split_cols: Callable, create_leaf: Callable, create_leaf_multi: Callable,
                          threshold: float, rdc_sample_size: int, next_operation: Optional[Callable] = None,
                          min_row_ratio: float = 0.01, rdc_strong_connection_threshold: float = 0.75,
                          multivariate_leaf: bool = True, initial_scope: Optional[List[int]] = None,
                          data_slicer: Callable = default_slicer, debug: bool = True):
    # 参数验证
    required_params = [dataset, ds_context, split_rows, split_cols, create_leaf, create_leaf_multi]
    if any(param is None for param in required_params):
        raise ValueError("必要参数不能为空")

    with managed_pool() as pool:
        # 初始化决策函数
        if next_operation is None:
            min_row = int(min_row_ratio * dataset.shape[0]) if min_row_ratio < 1 else int(min_row_ratio)
            next_operation = get_next_operation(
                ds_context, min_row, threshold=threshold, rdc_sample_size=rdc_sample_size,
                rdc_strong_connection_threshold=rdc_strong_connection_threshold,
                multivariate_leaf=multivariate_leaf
            )

        # 初始化根节点
        root = Independent()
        root.children.append(None)

        # 设置初始作用域
        initial_scope, initial_cond, num_conditional_cols = _setup_initial_scope_binary(dataset, initial_scope)

        # 任务队列处理
        tasks = deque()
        initial_task = (dataset, root, 0, initial_scope, initial_cond, None, 
                       False, False, False, False)
        tasks.append(initial_task)

        # 主处理循环
        _process_binary_tasks(tasks, next_operation, pool, create_leaf, create_leaf_multi,
                            data_slicer, num_conditional_cols, debug)

        # 完成并验证结果
        node = root.children[0]
        assign_ids(node)
        print(get_structure_stats(node))
        
        valid, err = is_valid(node)
        if not valid:
            raise ValueError(f"无效的SPN: {err}")
        
        node = Prune(node)
        valid, err = is_valid(node)
        if not valid:
            raise ValueError(f"修剪后的SPN无效: {err}")

        return node


def _setup_initial_scope_binary(dataset: np.ndarray, initial_scope: Optional[List[int]]):
    """设置二进制学习的初始作用域"""
    if initial_scope is None:
        return list(range(dataset.shape[1])), [], None
    
    if len(initial_scope) < dataset.shape[1]:
        num_conditional_cols = dataset.shape[1] - len(initial_scope)
        initial_cond = [i for i in range(dataset.shape[1]) if i not in initial_scope]
        return initial_scope, initial_cond, num_conditional_cols
    else:
        if len(initial_scope) > dataset.shape[1]:
            raise ValueError(f"初始作用域超出数据维度: {initial_scope}")
        return initial_scope, [], None


def _process_binary_tasks(tasks: deque, next_operation: Callable, pool, create_leaf: Callable,
                         create_leaf_multi: Callable, data_slicer: Callable, 
                         num_conditional_cols: Optional[int], debug: bool):
    """处理二进制学习的任务队列"""
    while tasks:
        task_params = tasks.popleft()
        local_data, parent, children_pos, scope, condition, rect_range = task_params[:6]
        no_clusters, no_independencies, no_condition, is_strong_connected = task_params[6:]

        if debug:
            logging.debug(f"处理任务: 数据{local_data.shape}, 作用域{scope}, 条件{condition}")

        # 数据验证
        validate_data_consistency(local_data.shape, scope, condition, "binary_task")

        # 获取操作
        operation, op_params = next_operation(
            local_data, scope, condition, no_clusters=no_clusters,
            no_independencies=no_independencies, no_condition=no_condition,
            is_strong_connected=is_strong_connected
        )

        if debug:
            logging.debug(f"操作: {operation} 数据形状: {local_data.shape} (剩余任务: {len(tasks)})")

        # 执行操作 - 这里需要实现具体的操作处理逻辑
        _execute_binary_operation(operation, op_params, local_data, parent, children_pos,
                                scope, condition, rect_range, no_clusters, no_independencies,
                                no_condition, is_strong_connected, tasks, pool, create_leaf,
                                create_leaf_multi, data_slicer, num_conditional_cols, debug)


def _execute_binary_operation(operation, op_params, local_data, parent, children_pos,
                            scope, condition, rect_range, no_clusters, no_independencies,
                            no_condition, is_strong_connected, tasks, pool, create_leaf,
                            create_leaf_multi, data_slicer, num_conditional_cols, debug):
    """执行二进制学习的具体操作"""
    
    if operation == Operation.CREATE_LEAF:
        start_time = perf_counter()
        
        if len(scope) == 1:
            node = create_leaf(local_data, ds_context, scope, condition)
        else:
            node = create_leaf_multi(local_data, ds_context, scope, condition)
        
        node.range = rect_range
        parent.children[children_pos] = node
        
        end_time = perf_counter()
        logging.debug(f"创建叶节点 {node.__class__.__name__} (作用域={scope}, 条件={condition}) "
                     f"耗时 {end_time - start_time:.5f}秒")