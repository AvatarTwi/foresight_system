import numpy as np
from sklearn.cluster import KMeans

from Learning.splitting.Base import split_data_by_clusters, clusters_by_adjacency_matrix
import logging

import itertools

from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_array
import scipy.stats

from sklearn.cross_decomposition import CCA
from Structure.StatisticalTypes import MetaType

logger = logging.getLogger(__name__)

CCA_MAX_ITER = 100


def ecdf(X):
    """
    Empirical cumulative distribution function
    for data X (one dimensional, if not it is linearized first)
    """
    if X.size == 0:
        return np.array([])
    
    mv_ids = np.isnan(X)
    N = X.shape[0]
    X_clean = X[~mv_ids]
    
    if X_clean.size == 0:
        return np.zeros(N)
    
    R = scipy.stats.rankdata(X_clean, method="max") / len(X_clean)
    X_r = np.zeros(N)
    X_r[~mv_ids] = R
    return X_r


def empirical_copula_transformation(data):
    """
    Apply empirical copula transformation to the data.
    Handle edge cases like empty arrays or 0-dimensional data.
    """
    if data.size == 0 or data.ndim == 0:
        return np.zeros((0, 2))
    
    transformed_data = np.apply_along_axis(ecdf, 0, data)
    ones_column = np.ones((data.shape[0], 1))
    return np.concatenate((transformed_data, ones_column), axis=1)


def make_matrix(data):
    """
    Ensures data to be 2-dimensional while handling empty arrays and sparse matrices
    """
    if hasattr(data, 'toarray'):
        return data
        
    if data.size == 0:
        return data.reshape(0, 1) if data.ndim == 1 else data

    if data.ndim == 1:
        return data[:, np.newaxis]
    elif data.ndim == 2:
        return data
    else:
        raise ValueError(f"Data must be 1 or 2 dimensional, got shape {data.shape}")


def _create_sparse_ohe(data, domain):
    """Helper function to create sparse one-hot encoding for large domains"""
    from scipy import sparse
    
    rows = np.arange(data.shape[0])
    cols = np.zeros(data.shape[0], dtype=int)
    
    # Create domain to index mapping for efficiency
    domain_to_idx = {val: idx for idx, val in enumerate(domain)}
    
    # Map data values to column indices
    for i, val in enumerate(data.ravel()):
        if not np.isnan(val) and val in domain_to_idx:
            cols[i] = domain_to_idx[val]
    
    # Create sparse matrix
    dataenc = sparse.csr_matrix(
        (np.ones(data.shape[0]), (rows, cols)),
        shape=(data.shape[0], len(domain))
    )
    
    # Handle NaN values
    nan_mask = np.isnan(data.ravel())
    if np.any(nan_mask):
        # Set NaN rows to all zeros
        dataenc[nan_mask] = 0
    
    return dataenc


def _create_dense_ohe(data, domain):
    """Helper function to create dense one-hot encoding for small domains"""
    dataenc = np.zeros((data.shape[0], len(domain)))
    dataenc[data[:, None] == domain[None, :]] = 1

    # Validation check
    if not np.any(np.isnan(data)):
        row_sums = np.nansum(dataenc, axis=1)
        if not np.all(row_sums == 1):
            raise ValueError(f"One hot encoding validation failed: some rows don't sum to 1")

    return dataenc


def ohe_data(data, domain, max_domain_size=10000):
    """
    One-hot encode data based on domain values.
    Efficiently handles large domains using sparse matrices.
    """
    if len(domain) > max_domain_size:
        logger.warning(f"Large domain detected: {len(domain)}, using sparse OHE")
        return _create_sparse_ohe(data, domain)
    else:
        return _create_dense_ohe(data, domain)


def _sample_large_domain(domain, max_size, rand_gen):
    """Helper function to sample from large domains"""
    if rand_gen is None:
        rand_gen = np.random.RandomState(17)
    
    sample_indices = rand_gen.choice(len(domain), size=max_size, replace=False)
    return domain[sample_indices]


def _process_discrete_feature(local_data, feature_idx, domain, max_domain_size, rand_gen):
    """Helper function to process discrete features with memory-efficient handling"""
    try:
        if len(domain) > max_domain_size:
            logger.warning(f"Feature {feature_idx} has large domain: {len(domain)}, using sparse OHE")
        
        feature_data = ohe_data(local_data[:, feature_idx], domain, max_domain_size)
        
        # Convert sparse to dense only if small enough
        if hasattr(feature_data, 'toarray') and feature_data.shape[1] <= max_domain_size:
            feature_data = feature_data.toarray()
            
        return feature_data
        
    except MemoryError:
        logger.warning(f"Memory error for feature {feature_idx}, sampling domain")
        sampled_domain = _sample_large_domain(domain, max_domain_size, rand_gen)
        feature_data = ohe_data(local_data[:, feature_idx], sampled_domain, max_domain_size)
        
        if hasattr(feature_data, 'toarray'):
            feature_data = feature_data.toarray()
            
        return feature_data


def _process_sparse_feature_chunks(sparse_matrix, max_domain_size, chunk_size=1000):
    """Process large sparse matrices in chunks to avoid memory issues"""
    n_chunks = (sparse_matrix.shape[1] + chunk_size - 1) // chunk_size
    result = np.zeros((sparse_matrix.shape[0], 2))
    
    for chunk_idx in range(n_chunks):
        start_col = chunk_idx * chunk_size
        end_col = min((chunk_idx + 1) * chunk_size, sparse_matrix.shape[1])
        
        chunk_data = sparse_matrix[:, start_col:end_col].toarray()
        chunk_sum = np.sum(chunk_data, axis=1, keepdims=True)
        result[:, 0] += chunk_sum.flatten()
    
    result[:, 1] = 1.0
    return result


def rdc_transformer(
    local_data,
    meta_types,
    domains,
    k=None,
    s=1.0 / 6.0,
    non_linearity=np.sin,
    return_matrix=False,
    ohe=True,
    rand_gen=None,
    max_domain_size=5000
):
    """
    Transform features according to the RDC pipeline:
    1 - empirical copula transformation
    2 - random projection into a k-dimensional gaussian space
    3 - pointwise non-linear transform
    """
    if local_data.size == 0:
        return np.zeros((0, 2)) if return_matrix else []

    N, D = local_data.shape

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    # Process features based on their meta types
    features = []
    for f in range(D):
        if meta_types[f] == MetaType.DISCRETE:
            feature_data = _process_discrete_feature(local_data, f, domains[f], max_domain_size, rand_gen)
            features.append(feature_data)
        else:
            features.append(local_data[:, f].reshape(-1, 1))

    # Set global k for all features
    if k is None:
        feature_shapes = [f.shape[1] if len(f.shape) > 1 else 1 for f in features]
        k = max(feature_shapes) + 1 if feature_shapes else 2

    # Process features to ensure proper format
    features_processed = []
    for f in features:
        if hasattr(f, 'toarray') and f.shape[1] > max_domain_size:
            # Process large sparse matrices in chunks
            processed_f = _process_sparse_feature_chunks(f, max_domain_size)
            features_processed.append(processed_f)
        else:
            # Convert smaller sparse matrices to dense
            if hasattr(f, 'toarray'):
                f = f.toarray()
            features_processed.append(make_matrix(f))

    features = features_processed

    # Apply empirical copula transformation
    features = [empirical_copula_transformation(f) for f in features]

    # Replace NaNs with zeros
    features = [np.nan_to_num(f) for f in features]

    # Random projection through gaussian
    random_gaussians = [rand_gen.normal(size=(f.shape[1], k)) for f in features]
    rand_proj_features = [s / f.shape[1] * np.dot(f, N) for f, N in zip(features, random_gaussians)]

    # Apply non-linearity
    nl_rand_proj_features = [non_linearity(f) for f in rand_proj_features]

    if return_matrix:
        return np.concatenate(nl_rand_proj_features, axis=1)
    else:
        return [np.concatenate((f, np.ones((f.shape[0], 1))), axis=1) for f in nl_rand_proj_features]


def rdc_cca(indexes):
    i, j, rdc_features = indexes
    cca = CCA(n_components=1, max_iter=CCA_MAX_ITER)
    X_cca, Y_cca = cca.fit_transform(rdc_features[i], rdc_features[j])
    rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    # logger.info(i, j, rdc)
    return rdc


def rdc_test(local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-1, rand_gen=None):
    n_features = local_data.shape[1]

    rdc_features = rdc_transformer(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, return_matrix=False, rand_gen=rand_gen
    )

    pairwise_comparisons = list(itertools.combinations(np.arange(n_features), 2))

    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
        delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
    )

    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    # Set diagonal to 1
    np.fill_diagonal(rdc_adjacency_matrix, 1)

    return rdc_adjacency_matrix


def getIndependentRDCGroups_py(
    local_data, threshold, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None
):
    rdc_adjacency_matrix = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
    )

    #
    # Why is this necessary?
    #
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    n_features = local_data.shape[1]

    #
    # thresholding
    rdc_adjacency_matrix[rdc_adjacency_matrix < threshold] = 0
    # logger.info("thresholding %s", rdc_adjacency_matrix)

    #
    # getting connected components
    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_array(rdc_adjacency_matrix))):
        result[list(c)] = i + 1

    return result


def _sample_domains_if_needed(domains, max_domain_size, rand_gen):
    """Helper function to sample large domains"""
    sampled_domains = []
    large_domains = []
    
    for i, domain in enumerate(domains):
        if len(domain) > max_domain_size:
            large_domains.append((i, len(domain)))
            sampled_domain = _sample_large_domain(domain, max_domain_size, rand_gen)
            sampled_domains.append(sampled_domain)
        else:
            sampled_domains.append(domain)
    
    if large_domains:
        logger.warning(f"Found large categorical domains: {large_domains}, which may cause memory issues")
        for idx, size in large_domains:
            logger.info(f"Sampling domain for feature {idx}, reducing from {size} to {max_domain_size}")
    
    return sampled_domains


def get_split_cols_RDC_py(threshold=0.3, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None,
                          max_sampling_threshold_cols=10000, max_domain_size=5000):
    def split_cols_RDC_py(local_data, ds_context, scope, clusters=None):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)
        
        # Sample large domains if needed
        if rand_gen is None:
            temp_rand_gen = np.random.RandomState(17)
        else:
            temp_rand_gen = rand_gen
            
        domains = _sample_domains_if_needed(domains, max_domain_size, temp_rand_gen)

        # Handle large datasets by sampling
        if local_data.shape[0] > max_sampling_threshold_cols:
            sample_indices = np.random.randint(local_data.shape[0], size=max_sampling_threshold_cols)
            local_data_sample = local_data[sample_indices, :]
        else:
            local_data_sample = local_data

        if clusters is None:
            clusters = getIndependentRDCGroups_py(
                local_data_sample,
                threshold,
                meta_types,
                domains,
                k=k,
                s=s,
                non_linearity=non_linearity,
                n_jobs=n_jobs,
                rand_gen=rand_gen,
            )
        
        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py


def get_split_rows_RDC_py(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None):
    def split_rows_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(
            local_data,
            meta_types,
            domains,
            k=k,
            s=s,
            non_linearity=non_linearity,
            return_matrix=True,
            rand_gen=rand_gen,
        )

        clusters = KMeans(n_clusters=n_clusters, random_state=rand_gen, n_jobs=n_jobs).fit_predict(rdc_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_RDC_py