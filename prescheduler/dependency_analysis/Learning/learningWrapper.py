from Learning.structureLearning import learn_structure
from Learning.structureLearning_binary import learn_structure_binary
from Learning.splitting.Condition_Clustering import *
from Learning.validity import is_valid
from Structure.nodes import Joint, assign_ids
from Structure.leaves.aspn_leaves.Multi_Histograms import create_multi_histogram_leaf
from Structure.leaves.aspn_leaves.Histograms import create_histogram_leaf
from Structure.leaves.binary.binary_leaf import create_binary_leaf
from Structure.leaves.binary.multi_binary_leaf import create_multi_binary_leaf
import itertools
import copy

import logging

logger = logging.getLogger(__name__)



def get_splitting_functions(cols, rows, ohe, threshold, rand_gen, n_jobs, max_sampling_threshold_rows=100000):
    from Learning.splitting.Clustering import get_split_rows_KMeans, get_split_rows_TSNE, get_split_rows_GMM
    if isinstance(cols, str):
        if cols == "rdc":
            from Learning.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs,
                                               max_sampling_threshold_cols=max_sampling_threshold_rows)
        elif cols == "poisson":
            from Learning.splitting.PoissonStabilityTest import get_split_cols_poisson_py
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
            split_rows_condition = None
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans(max_sampling_threshold_rows=max_sampling_threshold_rows)
            split_rows_condition = get_split_rows_condition_KMeans()
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
            split_rows_condition = get_split_rows_condition_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
            split_rows_condition = get_split_rows_condition_GMM()
        elif rows == "grid_naive":
            split_rows = get_split_rows_KMeans()
            split_rows_condition = get_split_rows_condition_Grid_naive()
        elif rows == "grid":
            split_rows = get_split_rows_KMeans(max_sampling_threshold_rows=max_sampling_threshold_rows)
            split_rows_condition = get_split_rows_condition_Grid()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows, split_rows_condition


class LearningConfig:
    """学习配置管理类"""
    def __init__(self, cols="rdc", rows="grid_naive", threshold=0.3, 
                 rdc_sample_size=50000, rdc_strong_connection_threshold=0.75,
                 multivariate_leaf=True, ohe=False, min_row_ratio=0.01, cpus=-1):
        self.cols = cols
        self.rows = rows
        self.threshold = threshold
        self.rdc_sample_size = rdc_sample_size
        self.rdc_strong_connection_threshold = rdc_strong_connection_threshold
        self.multivariate_leaf = multivariate_leaf
        self.ohe = ohe
        self.min_row_ratio = min_row_ratio
        self.cpus = cpus
    
    def validate(self):
        """验证配置参数的有效性"""
        assert 0 < self.threshold < 1, "阈值必须在0和1之间"
        assert self.rdc_sample_size > 0, "RDC样本大小必须大于0"
        assert 0 < self.rdc_strong_connection_threshold < 1, "强连接阈值必须在0和1之间"


def _setup_default_params(leaves, leaves_corr, rand_gen, config):
    """设置默认参数"""
    from Structure.leaves.aspn_leaves.Multi_Histograms import create_multi_histogram_leaf
    from Structure.leaves.aspn_leaves.Histograms import create_histogram_leaf
    
    if leaves is None:
        leaves = create_histogram_leaf
    if leaves_corr is None:
        leaves_corr = create_multi_histogram_leaf
    if rand_gen is None:
        rand_gen = np.random.RandomState(17)
    
    return leaves, leaves_corr, rand_gen


def learn_ASPN(data, ds_context, cols="rdc", rows="grid_naive", threshold=0.3,
               rdc_sample_size=50000, rdc_strong_connection_threshold=0.75,
               multivariate_leaf=True, ohe=False, leaves=None, leaves_corr=None,
               memory=None, rand_gen=None, cpus=-1):
    """
    优化后的ASPN学习函数
    """
    # 创建配置对象并验证
    config = LearningConfig(
        cols=cols, rows=rows, threshold=threshold,
        rdc_sample_size=rdc_sample_size,
        rdc_strong_connection_threshold=rdc_strong_connection_threshold,
        multivariate_leaf=multivariate_leaf, ohe=ohe, cpus=cpus
    )
    config.validate()
    
    # 设置默认参数
    leaves, leaves_corr, rand_gen = _setup_default_params(leaves, leaves_corr, rand_gen, config)

    def learn_param(data, ds_context, cols, rows, threshold, ohe):
        """内部学习参数函数"""
        split_cols, split_rows, split_rows_cond = get_splitting_functions(
            cols, rows, ohe, threshold, rand_gen, config.cpus, config.rdc_sample_size
        )

        return learn_structure(
            data, ds_context, split_rows, split_rows_cond, split_cols,
            leaves, leaves_corr, threshold=threshold,
            rdc_sample_size=config.rdc_sample_size,
            rdc_strong_connection_threshold=config.rdc_strong_connection_threshold,
            multivariate_leaf=config.multivariate_leaf
        )

    # 应用缓存（如果提供）
    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, config.cols, config.rows, config.threshold, config.ohe)


def learn_ASPN_binary(data, ds_context, cols="rdc", rows="grid_naive", threshold=0.3,
                      rdc_sample_size=50000, rdc_strong_connection_threshold=0.75,
                      multivariate_leaf=True, ohe=False, leaves=None, leaves_corr=None,
                      min_row_ratio=0.01, memory=None, rand_gen=None, cpus=-1):
    """
    优化后的二进制ASPN学习函数
    """
    # 创建配置对象
    config = LearningConfig(
        cols=cols, rows=rows, threshold=threshold,
        rdc_sample_size=rdc_sample_size,
        rdc_strong_connection_threshold=rdc_strong_connection_threshold,
        multivariate_leaf=multivariate_leaf, ohe=ohe,
        min_row_ratio=min_row_ratio, cpus=cpus
    )
    config.validate()
    
    # 设置二进制特定的默认参数
    from Structure.leaves.binary.binary_leaf import create_binary_leaf
    from Structure.leaves.binary.multi_binary_leaf import create_multi_binary_leaf
    
    if leaves is None:
        leaves = create_binary_leaf
    if leaves_corr is None:
        leaves_corr = create_multi_binary_leaf
    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, cols, rows, threshold, ohe):
        """内部学习参数函数"""
        split_cols, split_rows, split_rows_cond = get_splitting_functions(
            cols, rows, ohe, threshold, rand_gen, config.cpus, config.rdc_sample_size
        )

        return learn_structure_binary(
            data, ds_context, split_rows, split_rows_cond, split_cols,
            leaves, leaves_corr, threshold=threshold,
            rdc_sample_size=config.rdc_sample_size,
            rdc_strong_connection_threshold=config.rdc_strong_connection_threshold,
            min_row_ratio=config.min_row_ratio,
            multivariate_leaf=config.multivariate_leaf
        )

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, config.cols, config.rows, config.threshold, config.ohe)


def evidence_query_generate(data, data_true, query_ncol_max=3):
    """
    优化后的证据查询生成函数
    """
    nrow, ncol = data.shape
    
    # 参数验证
    assert query_ncol_max <= ncol, "查询列数不能超过总列数"
    assert nrow > 0 and ncol > 0, "数据不能为空"
    
    # 生成随机参数
    evidence_ncol = np.random.randint(2, max(3, ncol // 2))
    query_ncol = np.random.randint(1, min(query_ncol_max + 1, ncol - evidence_ncol + 1))
    
    # 选择列
    evidence_col = np.random.choice(ncol, size=evidence_ncol, replace=False)
    remaining_cols = [i for i in range(ncol) if i not in evidence_col]
    query_col = np.random.choice(remaining_cols, size=query_ncol, replace=False)

    # 生成查询组合
    query_options = [list(np.unique(data[:, i])) for i in query_col]
    query_combinations = list(itertools.product(*query_options))

    # 初始化查询边界
    num_queries = len(query_combinations)
    query_left = np.full((num_queries, ncol), -np.inf)
    query_right = np.full((num_queries, ncol), np.inf)

    # 设置证据约束
    data_subset = data_true.copy()
    for col_idx in evidence_col:
        evidence_value = data[np.random.randint(nrow), col_idx]
        query_left[:, col_idx] = evidence_value
        query_right[:, col_idx] = evidence_value
        data_subset = data_subset[data_subset[:, col_idx] == evidence_value]

    evidence_query = (query_left[0].copy(), query_right[0].copy())

    # 计算真实概率
    ground_truth = np.zeros(num_queries)
    total_samples = len(data_subset)
    
    if total_samples > 0:
        for i, query_values in enumerate(query_combinations):
            # 构建查询条件
            mask = np.ones(total_samples, dtype=bool)
            for j, col_idx in enumerate(query_col):
                query_left[i, col_idx] = query_values[j]
                query_right[i, col_idx] = query_values[j]
                mask &= (data_subset[:, col_idx] == query_values[j])
            
            ground_truth[i] = np.sum(mask) / total_samples

    return (query_left, query_right), evidence_query, ground_truth

