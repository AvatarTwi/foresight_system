import logging
import copy
import multiprocessing
import os
import time

from collections import deque
from enum import Enum
from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_array
from Learning.utils import convert_to_scope_domain, get_matached_domain
from Learning.statistics import get_structure_stats

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

from Learning.validity import is_valid
from Learning.splitting.RDC import rdc_test
from Structure.nodes import Independent, Joint, Decomposition, assign_ids

parallel = True

class PoolManager:
    """进程池管理器，确保资源正确释放"""
    def __init__(self):
        if parallel:
            cpus = max(1, os.cpu_count() - 6)
        else:
            cpus = 1
        self.pool = multiprocessing.Pool(processes=cpus)
    
    def __enter__(self):
        return self.pool
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()


def calculate_RDC(data, ds_context, scope, condition, sample_size):
    """
    使用数据计算RDC邻接矩阵
    """
    start_time = time.time()
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    meta_types = ds_context.get_meta_types_by_scope(scope_range)
    domains = ds_context.get_domains_by_scope(scope_range)

    # 根据数据大小选择采样策略
    if len(data) <= sample_size:
        sample_data = data
    else:
        sample_indices = np.random.randint(data.shape[0], size=sample_size)
        sample_data = data[sample_indices]
    
    rdc_adjacency_matrix = rdc_test(sample_data, meta_types, domains, k=10)
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    
    logging.debug(f"RDC计算耗时: {time.time() - start_time:.3f}秒, 样本数: {len(sample_data)}")
    return rdc_adjacency_matrix, scope_loc, condition_loc


class Operation(Enum):
    """
    操作类型枚举，定义了SPN学习过程中可能的操作
    """
    CREATE_LEAF = 1                # 创建叶节点
    SPLIT_COLUMNS = 2              # 创建乘积节点（基于列独立性）
    SPLIT_ROWS = 3                 # 创建求和节点（基于行聚类）
    NAIVE_FACTORIZATION = 4        # 简单假设所有变量独立
    REMOVE_UNINFORMATIVE_FEATURES = 5  # 移除无信息特征
    FACTORIZE = 6                  # 创建分解节点
    REMOVE_CONDITION = 7           # 从条件中移除独立变量
    SPLIT_ROWS_CONDITION = 8       # 带条件的行分割

    SPLIT_COLUMNS_CONDITION = 9    # 未实现: 带条件的列分割
    FACTORIZE_CONDITION = 10       # 未实现: 带条件的分解


def get_next_operation(ds_context, min_instances_slice=100, min_features_slice=1, multivariate_leaf=True,
                       threshold=0.3, rdc_sample_size=50000, rdc_strong_connection_threshold=0.75):
    """
    获取下一步操作的函数工厂
    
    参数:
        ds_context: 指定数据集中变量类型的上下文
        min_instances_slice: 停止行分割的最小行数
        min_features_slice: 停止列分割的最小特征数，通常为1
        multivariate_leaf: 如果为True，使用多变量联合分布拟合叶节点
        threshold: RDC相关性阈值
        rdc_sample_size: 计算RDC时的样本大小
        rdc_strong_connection_threshold: 强相关性阈值
        
    返回:
        用于确定下一步操作的函数
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

        query_attr_in_condition = [i for i in condition if i not in ds_context.fanout_attr]
        if is_strong_connected and (len(condition) == 0 or len(query_attr_in_condition) == 0):
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
    """
    默认的数据切片函数
    
    参数:
        data: 要切片的数据
        cols: 要保留的列索引
        num_cond_cols: 条件列的数量
        
    返回:
        切片后的数据
    """
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))

        return data[:, cols]
    else:
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)


def _validate_task_data(local_data, scope, condition):
    """验证任务数据的一致性"""
    expected_cols = len(scope) + len(condition)
    assert local_data.shape[1] == expected_cols, \
        f"数据列数不匹配: 期望{expected_cols}, 实际{local_data.shape[1]}"


def _create_operation_result(operation_type, node, parent, children_pos, **kwargs):
    """创建操作结果的标准化方法"""
    node.scope = kwargs.get('scope', [])
    node.condition = kwargs.get('condition', [])
    node.range = kwargs.get('rect_range')
    parent.children[children_pos] = node
    return node


def learn_structure(
        dataset,               # 输入数据集
        ds_context,            # 数据集上下文，包含特征类型信息
        split_rows,            # 按行分割数据的函数
        split_rows_condition,  # 带条件的按行分割函数
        split_cols,            # 按列分割数据的函数
        create_leaf,           # 创建叶节点的函数
        create_leaf_multi,     # 创建多变量叶节点的函数
        threshold,             # RDC相关性阈值
        rdc_sample_size,       # 计算RDC时的样本大小
        next_operation=None,   # 决定下一步操作的函数
        min_row_ratio=0.01,    # 最小行比例
        rdc_strong_connection_threshold=0.75,  # RDC强连接阈值
        multivariate_leaf=True,               # 是否使用多变量叶子
        create_leaf_fanout=None,              # 创建扇出叶节点的函数
        initial_scope=None,                   # 初始作用域
        data_slicer=default_slicer,           # 数据切片函数
        debug=True                            # 是否启用调试
):
    """
    学习SPN结构的主函数
    
    该函数通过递归地识别数据中的独立性和条件依赖关系，构建一个表示数据概率分布的分层结构模型。
    使用自顶向下的方法，不断划分数据并创建新节点，最终生成一个有效的SPN。
    
    参数:
        dataset: 要建模的数据集
        ds_context: 数据集上下文，包含各变量类型和域信息
        split_rows: 基于行聚类划分数据的函数
        split_rows_condition: 带条件的行聚类函数
        split_cols: 基于列独立性划分数据的函数
        create_leaf: 创建单变量叶节点的函数
        create_leaf_multi: 创建多变量叶节点的函数
        threshold: RDC相关性判断阈值
        rdc_sample_size: 计算RDC时使用的样本大小
        next_operation: 决定下一步操作的函数，如果为None则自动创建
        min_row_ratio: 最小行比例，用于确定min_instances_slice
        rdc_strong_connection_threshold: 强相关性阈值
        multivariate_leaf: 是否使用多变量叶节点
        create_leaf_fanout: 创建扇出叶节点的函数
        initial_scope: 初始作用域，如果为None则使用所有列
        data_slicer: 数据切片函数
        debug: 是否输出调试信息
        
    返回:
        构建好的SPN根节点
    """
    with PoolManager() as pool:
        # 参数验证
        required_params = [dataset, ds_context, split_rows, split_cols, create_leaf, create_leaf_multi]
        assert all(param is not None for param in required_params), "必要参数不能为空"

        # 初始化next_operation函数
        if next_operation is None:
            min_instances = int(min_row_ratio * dataset.shape[0]) if min_row_ratio < 1 else min_row_ratio
            next_operation = get_next_operation(
                ds_context, min_instances, threshold=threshold,
                rdc_sample_size=rdc_sample_size,
                rdc_strong_connection_threshold=rdc_strong_connection_threshold,
                multivariate_leaf=multivariate_leaf
            )

        # 初始化根节点和作用域
        root = Independent()
        root.children.append(None)

        if initial_scope is None:
            initial_scope = list(range(dataset.shape[1]))
            initial_cond = []
            num_conditional_cols = None
        else:
            if len(initial_scope) < dataset.shape[1]:
                num_conditional_cols = dataset.shape[1] - len(initial_scope)
                initial_cond = [i for i in range(dataset.shape[1]) if i not in initial_scope]
            else:
                num_conditional_cols = None
                initial_cond = []

        # 任务队列初始化
        tasks = deque()
        initial_task = (
            dataset, root, 0, initial_scope, initial_cond, None, None,
            False, False, False, False, None
        )
        tasks.append(initial_task)

        # 主处理循环
        while tasks:
            task_params = tasks.popleft()
            local_data, parent, children_pos, scope, condition = task_params[:5]
            cond_fanout_data, rect_range = task_params[5:7]
            no_clusters, no_independencies, no_condition, is_strong_connected, right_most_branch = task_params[7:]

            # 验证数据一致性
            _validate_task_data(local_data, scope, condition)

            # 确定下一步操作
            operation, op_params = next_operation(
                local_data, scope, condition,
                no_clusters=no_clusters, no_independencies=no_independencies,
                no_condition=no_condition, is_strong_connected=is_strong_connected
            )

            # 根据操作类型执行相应处理
            if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:
                # 移除无信息特征（方差为零的特征）
                (scope_rm, scope_rm2, scope_keep, condition_rm, condition_keep) = op_params
                new_condition = [condition[i] for i in condition_keep]
                keep_all = [item for item in range(local_data.shape[1]) if item not in condition_rm + scope_rm]

                if len(new_condition) != len(condition) and debug:
                    logging.debug(
                        f"find uninformation condition, keeping only condition {new_condition}")
                if len(scope_rm) == 0 and len(new_condition) != 0:
                    # only condition variables have been removed
                    assert (len(scope) + len(new_condition)) == len(
                        keep_all), f"Redundant data columns, {scope}, {new_condition}, {keep_all}"
                    tasks.append(
                        (
                            data_slicer(local_data, keep_all, num_conditional_cols),
                            parent,
                            children_pos,
                            scope,
                            new_condition,
                            cond_fanout_data,
                            rect_range,
                            no_clusters,
                            no_independencies,
                            True,
                            is_strong_connected,
                            right_most_branch
                        )
                    )
                    assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                        "node %s has same attribute in both condition and range"
                else:
                    # we need to create product node if scope variables have been removed
                    node = Independent()
                    node.scope = copy.deepcopy(scope)
                    node.condition = copy.deepcopy(new_condition)
                    node.range = copy.deepcopy(rect_range)
                    parent.children[children_pos] = node

                    rest_scope = copy.deepcopy(scope)
                    for i in range(len(scope_rm)):
                        col = scope_rm[i]
                        new_scope = scope[scope_rm2[i]]
                        rest_scope.remove(new_scope)
                        node.children.append(None)
                        assert col not in keep_all
                        if debug:
                            logging.debug(
                                f"find uninformative scope {new_scope}")
                        tasks.append(
                            (
                                data_slicer(local_data, [col], num_conditional_cols),
                                node,
                                len(node.children) - 1,
                                [new_scope],
                                [],
                                cond_fanout_data,
                                rect_range,
                                True,
                                True,
                                True,
                                False,
                                right_most_branch
                            )
                        )

                    next_final = False

                    if len(rest_scope) == 0:
                        continue
                    elif len(rest_scope) == 1:
                        next_final = True

                    node.children.append(None)
                    c_pos = len(node.children) - 1

                    if debug:
                        logging.debug(
                            f"The rest scope {rest_scope} and condition {new_condition} keep"
                        )
                        assert (len(rest_scope) + len(new_condition)) == len(keep_all), "Redundant data columns"
                    tasks.append(
                        (
                            data_slicer(local_data, keep_all, num_conditional_cols),
                            node,
                            c_pos,
                            rest_scope,
                            new_condition,
                            cond_fanout_data,
                            rect_range,
                            next_final,
                            next_final,
                            False,
                            is_strong_connected,
                            right_most_branch
                        )
                    )
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                continue

            elif operation == Operation.REMOVE_CONDITION:
                # 移除独立条件（与作用域无关的条件变量）
                (independent_condition, remove_cols) = op_params
                new_condition = [item for item in condition if item not in independent_condition]
                keep_cols = [item for item in range(local_data.shape[1]) if item not in remove_cols]
                if debug:
                    logging.debug(
                        f"Removed uniformative condition {independent_condition}")
                    assert (len(scope) + len(new_condition)) == len(keep_cols), "Redundant data columns"
                tasks.append(
                    (
                        data_slicer(local_data, keep_cols, num_conditional_cols),
                        parent,
                        children_pos,
                        scope,
                        new_condition,
                        cond_fanout_data,
                        rect_range,
                        no_clusters,
                        no_independencies,
                        True,
                        is_strong_connected,
                        right_most_branch
                    )
                )
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                continue

            elif operation == Operation.SPLIT_ROWS_CONDITION:
                # 带条件的行分割（聚类）
                query_attr = [i for i in condition if i not in ds_context.fanout_attr]
                if len(query_attr) == 0 and right_most_branch:
                    logging.debug(
                        f"\t\tcreate multi-leaves for scope {scope} and {condition}"
                    )
                    #if we only have fanout attr left in condition, there is no need to split by row
                    node = create_leaf_fanout(local_data, ds_context, scope, condition, cond_fanout_data)
                    node.range = rect_range
                    parent.children[children_pos] = node
                    continue

                split_start_t = perf_counter()
                data_slices = split_rows_condition(local_data, ds_context, scope, condition,
                                                op_params, cond_fanout_data=cond_fanout_data)
                split_end_t = perf_counter()

                if debug:
                    logging.debug(
                        "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                    )
                    if cond_fanout_data is not None:
                        assert len(local_data) == len(cond_fanout_data[1]), \
                        f"mismatched data length of {len(local_data)} and {len(cond_fanout_data[1])}"

                if len(data_slices) == 1:
                    tasks.append((local_data, parent, children_pos, scope, condition, cond_fanout_data,
                                rect_range, True, False, False, is_strong_connected, right_most_branch))
                    continue

                node = Joint()
                node.scope = copy.deepcopy(scope)
                node.condition = copy.deepcopy(condition)
                node.range = copy.deepcopy(rect_range)
                parent.children[children_pos] = node
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                for data_slice, range_slice, proportion, fanout_data_slice in data_slices:
                    assert (len(scope) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                    node.children.append(None)
                    node.weights.append(proportion)
                    new_rect_range = dict()
                    for c in rect_range:
                        if c not in range_slice:
                            new_rect_range[c] = rect_range[c]
                        else:
                            new_rect_range[c] = range_slice[c]
                    if debug and fanout_data_slice is not None:
                        assert len(data_slice) == len(fanout_data_slice[1]), \
                            f"mismatched data length of {len(data_slice)} and {len(fanout_data_slice[1])}"
                    tasks.append((data_slice, node, len(node.children) - 1, scope, condition,
                                fanout_data_slice, new_rect_range, False, False, False,
                                is_strong_connected, right_most_branch))
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                continue

            elif operation == Operation.SPLIT_ROWS:
                # 基于行聚类创建求和节点
                split_start_t = perf_counter()
                data_slices = split_rows(local_data, ds_context, scope)
                split_end_t = perf_counter()

                if debug:
                    logging.debug(
                        "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                    )

                if len(data_slices) == 1:
                    tasks.append((local_data, parent, children_pos, scope, condition, cond_fanout_data,
                                rect_range, False, True, False, is_strong_connected, right_most_branch))
                    continue

                node = Joint()
                node.scope = copy.deepcopy(scope)
                node.condition = copy.deepcopy(condition)
                node.range = copy.deepcopy(rect_range)
                parent.children[children_pos] = node
                node.cardinality = len(local_data)
                for data_slice, scope_slice, proportion, center in data_slices:
                    assert isinstance(scope_slice, list), "slice must be a list"
                    assert (len(scope) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                    node.children.append(None)
                    node.weights.append(proportion)
                    node.cluster_centers.append(center)
                    tasks.append((data_slice, node, len(node.children) - 1, scope, condition,
                                cond_fanout_data, rect_range, False, False, False,
                                is_strong_connected, right_most_branch))
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                continue

            elif operation == Operation.SPLIT_COLUMNS:
                # 基于列独立性创建乘积节点
                split_start_t = perf_counter()
                data_slices = split_cols(local_data, ds_context, scope, clusters=op_params)
                split_end_t = perf_counter()

                if debug:
                    logging.debug(
                        "\t\tfound {} col clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                    )

                if len(data_slices) == 1:
                    tasks.append((local_data, parent, children_pos, scope, condition, cond_fanout_data,
                                rect_range, False, True, False, is_strong_connected, right_most_branch))
                    assert np.shape(data_slices[0][0]) == np.shape(local_data)
                    assert data_slices[0][1] == scope
                    continue

                node = Independent()
                node.scope = copy.deepcopy(scope)
                node.condition = copy.deepcopy(condition)
                node.range = copy.deepcopy(rect_range)
                parent.children[children_pos] = node

                for data_slice, scope_slice, _ in data_slices:
                    assert isinstance(scope_slice, list), "slice must be a list"
                    assert (len(scope_slice) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                    node.children.append(None)
                    if debug:
                        logging.debug(
                            f'Create an independent component with scope {scope_slice} and condition {condition}'
                        )
                    tasks.append((data_slice, node, len(node.children) - 1, scope_slice, condition,
                                cond_fanout_data, rect_range, False, True, False,
                                is_strong_connected, right_most_branch))
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                continue

            elif operation == Operation.FACTORIZE:
                # 分解强连接组件
                node = Decomposition()
                node.scope = copy.deepcopy(scope)
                node.condition = copy.deepcopy(condition)
                node.range = copy.deepcopy(rect_range)
                parent.children[children_pos] = node
                index_list = sorted(scope + condition)

                # if there are multiple components we left it for the next round
                if debug:
                    for comp in op_params:
                        logging.debug(
                            f'Decomposition node found the strong connected component{comp}'
                        )
                    logging.debug(
                        f'We only factor out {op_params[0]}'
                    )

                strong_connected = op_params[0]
                other_connected = [item for item in scope if item not in strong_connected]

                assert len(other_connected) != 0, "factorize results in only one strongly connected"
                assert cond_fanout_data is None, "conditional data exists"
                node.children.append(None)
                data_copy = copy.deepcopy(local_data)
                if debug:
                    logging.debug(
                        f'Decomposition node factor out weak connected component{other_connected}'
                    )
                keep_cols = [index_list.index(i) for i in sorted(other_connected + condition)]
                tasks.append(
                    (
                        data_slicer(data_copy, keep_cols, num_conditional_cols),
                        node,
                        0,
                        other_connected,
                        condition,
                        None,
                        rect_range,
                        False,
                        False,
                        False,
                        False,
                        False
                    )
                )
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                new_condition = sorted(condition + other_connected)
                node.children.append(None)
                new_scope = strong_connected
                keep_cols = [index_list.index(i) for i in sorted(new_scope + new_condition)]
                if debug:
                    logging.debug(
                        f'Decomposition node found a strongly connect component{new_scope}, '
                        f'condition on {new_condition}'
                    )
                    assert (len(new_scope) + len(new_condition)) == len(keep_cols), "Redundant data columns"
                if rect_range is None:
                    new_rect_range = dict()
                else:
                    new_rect_range = copy.deepcopy(rect_range)
                for i, c in enumerate(new_condition):
                    condition_idx = []
                    for j in new_condition:
                        condition_idx.append(index_list.index(j))
                    data_attr = local_data[:, condition_idx[i]]
                    new_rect_range[c] = [(np.nanmin(data_attr), np.nanmax(data_attr))]
                cond_fanout_attr = [i for i in new_condition if i in ds_context.fanout_attr]
                if len(cond_fanout_attr) == 0:
                    new_condition_fanout_data = None
                else:
                    cond_fanout_keep_cols = [index_list.index(i) for i in cond_fanout_attr]
                    new_condition_fanout_data = (cond_fanout_attr,
                                                data_slicer(local_data, cond_fanout_keep_cols, num_conditional_cols))
                if right_most_branch is None:
                    new_right_most_branch = True
                else:
                    new_right_most_branch = False
                tasks.append(
                    (
                        data_slicer(local_data, keep_cols, num_conditional_cols),
                        node,
                        1,
                        new_scope,
                        new_condition,
                        new_condition_fanout_data,
                        new_rect_range,
                        False,
                        True,
                        False,
                        True,
                        new_right_most_branch
                    )
                )
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                continue

            elif operation == Operation.NAIVE_FACTORIZATION:
                # 假设变量独立，创建乘积节点并为每个变量创建单独的叶节点
                node = Independent()
                node.scope = copy.deepcopy(scope)
                node.condition = copy.deepcopy(condition)
                node.range = copy.deepcopy(rect_range)
                parent.children[children_pos] = node

                scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
                local_tasks = []
                local_children_params = []
                split_start_t = perf_counter()
                for i, col in enumerate(scope_loc):
                    node.children.append(None)
                    local_tasks.append(len(node.children) - 1)
                    child_data_slice = data_slicer(local_data, [col], num_conditional_cols)
                    local_children_params.append((child_data_slice, ds_context, [scope[i]], []))

                result_nodes = pool.starmap(create_leaf, local_children_params)

                for child_pos, child in zip(local_tasks, result_nodes):
                    node.children[child_pos] = child

                split_end_t = perf_counter()

                logging.debug(
                    "\t\tnaive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t)
                )
                continue

            elif operation == Operation.CREATE_LEAF:
                # 创建叶节点
                start_time = perf_counter()
                
                # 选择合适的叶节点创建函数
                if (cond_fanout_data is None or len(cond_fanout_data) == 0) and len(scope) == 1:
                    node = create_leaf(local_data, ds_context, scope, condition)
                elif create_leaf_fanout is None:
                    node = create_leaf_multi(local_data, ds_context, scope, condition)
                elif right_most_branch:
                    node = create_leaf_fanout(local_data, ds_context, scope, condition, cond_fanout_data)
                else:
                    # 处理混合情况
                    curr_fanout_attr = [i for i in scope + condition if i in ds_context.fanout_attr]
                    if cond_fanout_data is None and len(curr_fanout_attr) == 0:
                        node = create_leaf_multi(local_data, ds_context, scope, condition)
                    else:
                        prob_mhl = create_leaf_multi(local_data, ds_context, scope, condition)
                        exp_mhl = create_leaf_fanout(local_data, ds_context, scope, condition, cond_fanout_data)
                        node = Multi_histogram_full(prob_mhl, exp_mhl, scope)
                
                node.range = rect_range
                node.cardinality = len(local_data)
                parent.children[children_pos] = node
                
                end_time = perf_counter()
                
                # 验证节点有效性
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "父节点的条件和作用域存在重叠"


        # 获取并验证最终结果
        final_node = root.children[0]
        assign_ids(final_node)
        
        print(get_structure_stats(final_node))
        
        is_valid_spn, error_msg = is_valid(final_node)
        assert is_valid_spn, f"生成的SPN无效: {error_msg}"

        return final_node

