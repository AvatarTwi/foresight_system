from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import logging

from Learning.splitting.Base import split_data_by_clusters, preproc
from Learning.splitting.Rect_approaximate import rect_approximate
from Learning.splitting.Grid_clustering import get_optimal_attribute, get_optimal_split_naive, get_optimal_split

logger = logging.getLogger(__name__)


def _extract_condition_data(data, scope, condition):
    """Helper function to extract conditioning data indices"""
    range_idx = sorted(scope + condition)
    condition_idx = [range_idx.index(i) for i in condition]
    return condition_idx


def _validate_clustering_inputs(local_data, ds_context, scope, condition):
    """Validate inputs for clustering operations"""
    if local_data.size == 0:
        raise ValueError("Local data is empty")
    
    if not condition:
        raise ValueError("Condition list is empty")
    
    if not scope:
        raise ValueError("Scope list is empty")
    
    return True


def get_split_rows_condition_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    """
    Create KMeans clustering function for conditioning variables
    """
    def split_rows_KMeans(local_data, ds_context, scope, condition, rdc_mat=None):
        try:
            _validate_clustering_inputs(local_data, ds_context, scope, condition)
            
            # Preprocess data
            data = preproc(local_data, ds_context, pre_proc, ohe)
            
            # Extract conditioning columns
            condition_idx = _extract_condition_data(data, scope, condition)
            
            if not condition_idx:
                raise ValueError("No valid conditioning indices found")
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            clusters = kmeans.fit_predict(data[:, condition_idx])
            
            return split_data_by_clusters(local_data, clusters, scope, rows=True)
            
        except Exception as e:
            logger.error(f"KMeans clustering failed: {e}")
            # Fallback: return single cluster
            return [(local_data, scope, 1.0)]
    
    return split_rows_KMeans


def get_split_rows_condition_TSNE(n_clusters=2, pre_proc=None, ohe=False, seed=17, 
                                  verbose=0, n_jobs=-1):
    """
    Create t-SNE + KMeans clustering function with improved error handling
    """
    try:
        from MulticoreTSNE import MulticoreTSNE as TSNE
    except ImportError:
        logger.error("MulticoreTSNE not available, falling back to regular KMeans")
        return get_split_rows_condition_KMeans(n_clusters, pre_proc, ohe, seed)
    
    # Determine number of CPUs
    ncpus = n_jobs if n_jobs > 0 else max(os.cpu_count() - 1, 1)
    
    def split_rows_TSNE_KMeans(local_data, ds_context, scope, condition, rdc_mat=None):
        try:
            _validate_clustering_inputs(local_data, ds_context, scope, condition)
            
            # Preprocess data
            data = preproc(local_data, ds_context, pre_proc, ohe)
            condition_idx = _extract_condition_data(data, scope, condition)
            cond_data = data[:, condition_idx]
            
            # Check if data is suitable for t-SNE
            if cond_data.shape[0] < 4 or cond_data.shape[1] < 2:
                logger.warning("Data too small for t-SNE, using regular KMeans")
                clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(cond_data)
            else:
                # Apply t-SNE dimensionality reduction
                n_components = min(3, cond_data.shape[1])
                tsne = TSNE(n_components=n_components, verbose=verbose, 
                           n_jobs=ncpus, random_state=seed, perplexity=min(30, cond_data.shape[0]-1))
                
                tsne_data = tsne.fit_transform(cond_data)
                
                # Apply KMeans to t-SNE results
                clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(tsne_data)
            
            return split_data_by_clusters(local_data, clusters, scope, rows=True)
            
        except Exception as e:
            logger.error(f"t-SNE clustering failed: {e}")
            # Fallback to regular KMeans
            fallback_func = get_split_rows_condition_KMeans(n_clusters, pre_proc, ohe, seed)
            return fallback_func(local_data, ds_context, scope, condition, rdc_mat)
    
    return split_rows_TSNE_KMeans


def get_split_rows_condition_DBScan(eps=2, min_samples=10, pre_proc=None, ohe=False):
    """
    Create DBSCAN clustering function with improved parameter handling
    """
    def split_rows_DBScan(local_data, ds_context, scope, condition, rdc_mat=None):
        try:
            _validate_clustering_inputs(local_data, ds_context, scope, condition)
            
            # Preprocess data
            data = preproc(local_data, ds_context, pre_proc, ohe)
            condition_idx = _extract_condition_data(data, scope, condition)
            
            # Adaptive parameter adjustment
            n_samples = data.shape[0]
            adaptive_min_samples = min(min_samples, max(2, n_samples // 100))
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=adaptive_min_samples)
            clusters = dbscan.fit_predict(data[:, condition_idx])
            
            # Handle case where DBSCAN returns all noise points (-1)
            if np.all(clusters == -1):
                logger.warning("DBSCAN found only noise points, using single cluster")
                clusters = np.zeros(len(clusters))
            
            # Reassign cluster labels to be non-negative
            unique_clusters = np.unique(clusters)
            cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
            clusters = np.array([cluster_mapping[c] for c in clusters])
            
            return split_data_by_clusters(local_data, clusters, scope, rows=True)
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return [(local_data, scope, 1.0)]
    
    return split_rows_DBScan


def get_split_rows_condition_GMM(n_clusters=2, pre_proc=None, ohe=False, seed=17, 
                                max_iter=100, n_init=2, covariance_type="full"):
    """
    Create Gaussian Mixture Model clustering function with validation
    """
    valid_covariance_types = ['spherical', 'diag', 'tied', 'full']
    if covariance_type not in valid_covariance_types:
        logger.warning(f"Invalid covariance type {covariance_type}, using 'full'")
        covariance_type = "full"
    
    def split_rows_GMM(local_data, ds_context, scope, condition, rdc_mat=None):
        try:
            _validate_clustering_inputs(local_data, ds_context, scope, condition)
            
            # Preprocess data
            data = preproc(local_data, ds_context, pre_proc, ohe)
            condition_idx = _extract_condition_data(data, scope, condition)
            cond_data = data[:, condition_idx]
            
            # Check if we have enough samples for GMM
            if cond_data.shape[0] < n_clusters * 2:
                logger.warning(f"Too few samples ({cond_data.shape[0]}) for {n_clusters} clusters")
                n_clusters_adjusted = max(1, cond_data.shape[0] // 2)
            else:
                n_clusters_adjusted = n_clusters
            
            # Fit Gaussian Mixture Model
            estimator = GaussianMixture(
                n_components=n_clusters_adjusted,
                covariance_type=covariance_type,
                max_iter=max_iter,
                n_init=n_init,
                random_state=seed,
                init_params='kmeans'  # More stable initialization
            )
            
            estimator.fit(cond_data)
            clusters = estimator.predict(cond_data)
            
            return split_data_by_clusters(local_data, clusters, scope, rows=True)
            
        except Exception as e:
            logger.error(f"GMM clustering failed: {e}")
            # Fallback to KMeans
            fallback_func = get_split_rows_condition_KMeans(n_clusters, pre_proc, ohe, seed)
            return fallback_func(local_data, ds_context, scope, condition, rdc_mat)
    
    return split_rows_GMM


def _get_fanout_attribute_locations(scope, condition, ds_context):
    """Extract fanout attribute locations"""
    fanout_attr = [i for i in condition if i in getattr(ds_context, 'fanout_attr', [])]
    idx_range = sorted(scope + condition)
    return [idx_range.index(i) for i in fanout_attr]


def get_split_rows_condition_Grid_naive(n_clusters=2, pre_proc=None, ohe=False):
    """
    Create naive grid-based clustering function
    """
    def split_rows_Grid(local_data, ds_context, scope, condition, rdc_mat=None, cond_fanout_data=None):
        try:
            _validate_clustering_inputs(local_data, ds_context, scope, condition)
            
            # Preprocess data
            data = preproc(local_data, ds_context, pre_proc, ohe)
            
            # Get fanout attribute locations
            fanout_attr_loc = _get_fanout_attribute_locations(scope, condition, ds_context)
            
            # Find optimal attribute for splitting
            opt_attr, opt_attr_idx = get_optimal_attribute(rdc_mat, fanout_attr=fanout_attr_loc)
            
            logger.info(f"Optimal attribute found: {condition[opt_attr_idx]}")
            
            # Get clusters and ranges
            attr_type = getattr(ds_context, 'meta_types', {})[condition[opt_attr_idx]]
            clusters, range_slice = get_optimal_split_naive(
                data[:, opt_attr], condition[opt_attr_idx], attr_type, n_clusters
            )
            
            # Split data
            temp_res = split_data_by_clusters(local_data, clusters, scope, rows=True)
            
            # Handle fanout data if provided
            if cond_fanout_data is not None:
                if len(cond_fanout_data) != 2:
                    raise ValueError("Incorrect shape for conditional fanout data")
                if len(cond_fanout_data[1]) != len(data):
                    raise ValueError("Mismatched data length")
                
                fanout_res = split_data_by_clusters(cond_fanout_data[1], clusters, scope, rows=True)
                
                # Combine results with fanout information
                res = []
                for i, (data_slice, scope_slice, proportion) in enumerate(temp_res):
                    fanout_info = (cond_fanout_data[0], fanout_res[i][0])
                    res.append((data_slice, range_slice[i], proportion, fanout_info))
            else:
                # Standard results without fanout
                res = []
                for i, (data_slice, scope_slice, proportion) in enumerate(temp_res):
                    res.append((data_slice, range_slice[i], proportion, None))
            
            return res
            
        except Exception as e:
            logger.error(f"Naive grid clustering failed: {e}")
            return [(local_data, {}, 1.0, None)]
    
    return split_rows_Grid


def get_split_rows_condition_Grid(n_clusters=2, pre_proc=None, ohe=False, 
                                 eval_func=np.max, seed=17):
    """
    Create advanced grid-based clustering function
    """
    def split_rows_Grid(local_data, ds_context, scope, condition, rdc_mat=None):
        try:
            _validate_clustering_inputs(local_data, ds_context, scope, condition)
            
            # Preprocess data
            data = preproc(local_data, ds_context, pre_proc, ohe)
            
            # Get fanout attribute locations  
            fanout_attr_loc = _get_fanout_attribute_locations(scope, condition, ds_context)
            
            # Find optimal attribute and split
            opt_attr, opt_attr_idx = get_optimal_attribute(rdc_mat, fanout_attr=fanout_attr_loc)
            
            clusters, range_slice = get_optimal_split(
                data, ds_context, scope, condition, opt_attr,
                condition[opt_attr_idx], n_clusters, eval_func=eval_func
            )
            
            # Split and return results
            temp_res = split_data_by_clusters(local_data, clusters, scope, rows=True)
            
            res = []
            for i, (data_slice, scope_slice, proportion) in enumerate(temp_res):
                res.append((data_slice, range_slice[i], proportion))
            
            return res
            
        except Exception as e:
            logger.error(f"Grid clustering failed: {e}")
            return [(local_data, {}, 1.0)]
    
    return split_rows_Grid


def get_split_rows_condition_Rect(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    """
    Create rectangle approximation clustering function
    """
    def split_rows_Rect(local_data, ds_context, scope, condition, rdc_mat=None):
        try:
            _validate_clustering_inputs(local_data, ds_context, scope, condition)
            
            # Preprocess data
            data = preproc(local_data, ds_context, pre_proc, ohe)
            condition_idx = _extract_condition_data(data, scope, condition)
            
            # Initial KMeans clustering
            initial_clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data[:, condition_idx])
            
            # Rectangle approximation
            clusters, range_slice = rect_approximate(data[:, condition_idx], initial_clusters)
            
            # Split and return results
            temp_res = split_data_by_clusters(local_data, clusters, scope, rows=True)
            
            res = []
            for i, (data_slice, scope_slice, proportion) in enumerate(temp_res):
                res.append((data_slice, range_slice[i], proportion))
            
            return res
            
        except Exception as e:
            logger.error(f"Rectangle approximation clustering failed: {e}")
            return [(local_data, {}, 1.0)]
    
    return split_rows_Rect
