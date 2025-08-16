import numpy as np
import logging
from Calculate.inference import EPSILON
from Learning.splitting.RDC import rdc_cca, rdc_transformer
from Learning.utils import convert_to_scope_domain
from Structure.StatisticalTypes import MetaType

logger = logging.getLogger(__name__)


def get_optimal_attribute(rdc_op, eval_func=np.max, fanout_attr=None):
    """
    Select optimal attributes to split on using pairwise RDC matrix
    """
    if fanout_attr is None:
        fanout_attr = []
    
    rdc_mat, scope_loc, condition_loc = rdc_op
    
    # Determine query attributes, excluding fanout attributes if needed
    if not fanout_attr or len(fanout_attr) == len(condition_loc):
        query_attr_loc = condition_loc
    else:
        query_attr_loc = [i for i in condition_loc if i not in fanout_attr]
    
    logger.debug(f"fanout_location {fanout_attr}, condition_location {condition_loc}, "
                 f"query_attr {query_attr_loc}")

    if not query_attr_loc:
        raise ValueError("No valid attributes to select from")

    corr_min = float('inf')
    opt_attr = None
    
    for c in query_attr_loc:
        # Vectorized computation of RDC values
        rdc_vals = np.array([rdc_mat[c][s] for s in scope_loc])
        corr = eval_func(rdc_vals)
        
        if corr < corr_min:
            corr_min = corr
            opt_attr = c
    
    if opt_attr is None:
        raise ValueError("No optimal attribute found")
    
    return opt_attr, condition_loc.index(opt_attr)


def _create_range_dict(attr, value_range):
    """Helper function to create range dictionary"""
    return {attr: value_range}


def _get_binary_clusters(data, attr):
    """Handle binary data clustering"""
    clusters = np.zeros(len(data))
    rect_range = []
    
    clusters[data == 0] = 0
    rect_range.append(_create_range_dict(attr, [0]))
    
    clusters[data == 1] = 1
    rect_range.append(_create_range_dict(attr, [1]))
    
    return clusters, rect_range


def _get_unique_value_clusters(data, attr, unique_vals, n_clusters):
    """Handle clustering when unique values <= n_clusters"""
    clusters = np.zeros(len(data))
    rect_range = []
    
    for i, val in enumerate(unique_vals):
        mask = data == val
        clusters[mask] = i
        rect_range.append(_create_range_dict(attr, [(val, val)]))
    
    return clusters, rect_range


def _get_median_split_clusters(data, attr):
    """Handle median-based binary split"""
    clusters = np.zeros(len(data))
    median_val = np.nanmedian(data)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    if median_val == min_val:
        # Handle case where median equals minimum
        clusters[data <= median_val] = 0
        clusters[data > median_val] = 1
        
        right_min = np.nanmin(data[data > median_val]) if np.any(data > median_val) else max_val
        rect_range = [
            _create_range_dict(attr, [(min_val, median_val)]),
            _create_range_dict(attr, [(right_min, max_val)])
        ]
    else:
        # Standard median split
        clusters[data < median_val] = 0
        clusters[data >= median_val] = 1
        
        left_max = np.nanmax(data[data < median_val])
        rect_range = [
            _create_range_dict(attr, [(min_val, left_max)]),
            _create_range_dict(attr, [(median_val, max_val)])
        ]
    
    return clusters, rect_range


def _get_histogram_clusters(data, attr, n_clusters):
    """Handle histogram-based clustering"""
    clusters = np.zeros(len(data))
    
    try:
        density, breaks = np.histogram(data, bins=n_clusters)
        rect_range = []
        
        for i in range(len(density)):
            # Find data points in this bin
            if i == len(density) - 1:
                # Last bin includes right edge
                mask = (data >= breaks[i]) & (data <= breaks[i + 1])
            else:
                mask = (data >= breaks[i]) & (data < breaks[i + 1])
            
            clusters[mask] = i
            rect_range.append(_create_range_dict(attr, [(breaks[i] + EPSILON, breaks[i + 1])]))
        
        return clusters, rect_range
    
    except Exception as e:
        logger.warning(f"Histogram clustering failed: {e}, falling back to median split")
        return _get_median_split_clusters(data, attr)


def get_optimal_split_naive(data, attr, attr_type, n_clusters):
    """
    Split attribute naively based on data characteristics
    """
    if len(data) == 0:
        return np.array([]), []
    
    # Handle binary attributes
    if attr_type == MetaType.BINARY:
        return _get_binary_clusters(data, attr)
    
    # Get unique values
    unique_vals = np.unique(data[~np.isnan(data)])
    
    # Handle case with few unique values
    if len(unique_vals) <= n_clusters:
        return _get_unique_value_clusters(data, attr, unique_vals, n_clusters)
    
    # Handle binary split
    if n_clusters == 2:
        return _get_median_split_clusters(data, attr)
    
    # Handle multi-way split with histogram
    return _get_histogram_clusters(data, attr, n_clusters)


def _validate_rdc_inputs(local_data, ds_context, scope, condition, attr_loc):
    """Validate inputs for RDC computation"""
    if local_data.size == 0:
        raise ValueError("Local data is empty")
    
    if attr_loc >= local_data.shape[1]:
        raise ValueError(f"attr_loc {attr_loc} exceeds data dimensions")
    
    return True


def sub_range_rdc_test(local_data, ds_context, scope, condition, attr_loc, rdc_sample=50000):
    """
    Compute RDC test for data subrange with improved error handling
    """
    _validate_rdc_inputs(local_data, ds_context, scope, condition, attr_loc)
    
    # Sample data if necessary
    if len(local_data) <= rdc_sample:
        data_sample = local_data
    else:
        sample_indices = np.random.randint(local_data.shape[0], size=rdc_sample)
        data_sample = local_data[sample_indices]
    
    try:
        scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
        meta_types = ds_context.get_meta_types_by_scope(scope_range)
        domains = ds_context.get_domains_by_scope(scope_range)
        
        rdc_features = rdc_transformer(
            data_sample, meta_types, domains, k=10, s=1.0 / 6.0, 
            non_linearity=np.sin, return_matrix=False, rand_gen=None
        )
        
        from joblib import Parallel, delayed
        
        rdc_vals = Parallel(n_jobs=-1, max_nbytes=1024, backend="threading")(
            delayed(rdc_cca)((i, attr_loc, rdc_features)) for i in scope_loc
        )
        
        rdc_vector = np.array(rdc_vals)
        rdc_vector[np.isnan(rdc_vector)] = 0
        
        return rdc_vector
        
    except Exception as e:
        logger.error(f"RDC computation failed: {e}")
        return np.zeros(len(scope))


def get_equal_width_binning(local_data, num_bins, min_threshold_ratio=1/10000):
    """
    Create equal-width bins with improved logic
    """
    if len(local_data) == 0:
        return []
    
    n = len(local_data)
    min_threshold = n * min_threshold_ratio
    categories = sorted(np.unique(local_data))
    
    if len(categories) <= num_bins:
        return categories
    
    bin_freq = 1.0 / num_bins
    freq = 0.0
    bins = []
    
    for i, category in enumerate(categories):
        freq += np.sum(local_data == category) / n
        
        # Add bin if frequency threshold reached or at the end
        if freq >= bin_freq or i == len(categories) - 1:
            if freq * n > min_threshold or i == len(categories) - 1:
                bins.append(category)
            freq = 0.0
    
    return bins if bins else [categories[-1]]


def get_optimal_split(data, ds_context, scope, condition, attr_loc, attr, n_clusters=2,
                      rdc_sample=100000, eval_func=np.max, num_bins=15):
    """
    Split attribute based on pairwise RDC values with optimization
    """
    if n_clusters != 2:
        raise ValueError("Only dichotomy supported. Use recursively for multiple clusters")
    
    if data.size == 0:
        return np.array([]), []
    
    data_attr = data[:, attr_loc]
    unique_vals = np.unique(data_attr[~np.isnan(data_attr)])
    
    # Handle simple cases
    if len(unique_vals) <= 2:
        clusters = np.zeros(len(data_attr))
        rect_range = []
        
        for i, val in enumerate(unique_vals):
            mask = data_attr == val
            clusters[mask] = i
            rect_range.append(_create_range_dict(attr, [(val, val + EPSILON)]))
        
        return clusters, rect_range
    
    # Find optimal split point
    bins = get_equal_width_binning(data_attr, num_bins)
    
    if not bins:
        logger.warning("No valid bins found, using median split")
        return _get_median_split_clusters(data_attr, attr)
    
    best_score = float('inf')
    best_split = None
    
    for split_val in bins:
        # Create data splits
        left_mask = data_attr <= split_val
        right_mask = data_attr > split_val
        
        if np.sum(left_mask) < 10 or np.sum(right_mask) < 10:
            continue
        
        try:
            # Compute RDC for each split
            data_left = data[left_mask]
            data_right = data[right_mask]
            
            rdc_left = sub_range_rdc_test(data_left, ds_context, scope, condition, attr_loc, rdc_sample)
            rdc_right = sub_range_rdc_test(data_right, ds_context, scope, condition, attr_loc, rdc_sample)
            
            score = eval_func(rdc_left) + eval_func(rdc_right)
            
            if score < best_score:
                best_score = score
                best_split = split_val
                
        except Exception as e:
            logger.warning(f"Error evaluating split at {split_val}: {e}")
            continue
    
    # Create final clusters
    if best_split is None:
        logger.warning("No valid split found, using median")
        return _get_median_split_clusters(data_attr, attr)
    
    clusters = np.zeros(len(data_attr))
    clusters[data_attr <= best_split] = 0
    clusters[data_attr > best_split] = 1
    
    min_val = np.nanmin(data_attr)
    max_val = np.nanmax(data_attr)
    right_min = np.nanmin(data_attr[data_attr > best_split]) if np.any(data_attr > best_split) else max_val
    
    rect_range = [
        _create_range_dict(attr, [(min_val, best_split)]),
        _create_range_dict(attr, [(right_min, max_val)])
    ]
    
    logger.info(f"Optimal clusters found: {rect_range}")
    return clusters, rect_range
