import numpy as np
from Learning.splitting.Base import split_data_by_clusters
import logging

logger = logging.getLogger(__name__)


def _validate_data_for_conditioning(local_data):
    """Validate input data for conditioning operations"""
    if local_data.size == 0:
        raise ValueError("Input data is empty")
    
    if local_data.ndim != 2:
        raise ValueError(f"Data must be 2-dimensional, got {local_data.ndim}")
    
    return local_data.shape


def _find_valid_conditioning_columns(local_data):
    """Find columns suitable for conditioning (not all 0s or all 1s)"""
    n_rows, n_cols = local_data.shape
    valid_columns = []
    
    for col in range(n_cols):
        col_data = local_data[:, col]
        ones_count = np.sum(col_data)
        
        # Column is valid if it has both 0s and 1s
        if 0 < ones_count < n_rows:
            valid_columns.append(col)
    
    return valid_columns


def get_split_rows_random_conditioning():
    """
    Create a function for random conditioning with improved validation
    """
    def split_rows_random_conditioning(local_data):
        try:
            _validate_data_for_conditioning(local_data)
            
            # Find valid columns for conditioning
            valid_columns = _find_valid_conditioning_columns(local_data)
            
            if not valid_columns:
                logger.warning("No valid columns found for conditioning")
                return None, False
            
            # Randomly choose from valid columns
            choice = np.random.choice(valid_columns)
            return choice, True
            
        except Exception as e:
            logger.error(f"Random conditioning failed: {e}")
            return None, False
    
    return split_rows_random_conditioning


def _compute_column_statistics(col_data, alpha=0.1):
    """Compute statistics for a single column"""
    n_samples = len(col_data)
    ones_count = np.sum(col_data)
    zeros_count = n_samples - ones_count
    
    # Smoothed probability estimate
    prob = (ones_count + alpha) / (n_samples + 2 * alpha)
    
    return ones_count, zeros_count, prob


def naive_ll(local_data, alpha=0.1):
    """
    Compute naive log-likelihood with improved numerical stability
    """
    if local_data.size == 0:
        return -np.inf
    
    n_rows, n_cols = local_data.shape
    total_ll = 0.0
    
    for col in range(n_cols):
        col_data = local_data[:, col]
        ones_count, zeros_count, prob = _compute_column_statistics(col_data, alpha)
        
        # Compute log probabilities with numerical stability
        if prob <= 0 or prob >= 1:
            logger.warning(f"Invalid probability {prob} for column {col}")
            continue
        
        log_prob_one = np.log(prob)
        log_prob_zero = np.log(1 - prob)
        
        # Accumulate log-likelihood
        col_ll = ones_count * log_prob_one + zeros_count * log_prob_zero
        total_ll += col_ll
    
    return total_ll / n_rows if n_rows > 0 else -np.inf


def _evaluate_conditioning_split(local_data, col_conditioning, scope):
    """Evaluate the quality of a conditioning split"""
    try:
        # Create binary clusters based on conditioning column
        conditioning_values = local_data[:, col_conditioning]
        clusters = (conditioning_values == 1).astype(int)
        
        # Split data into clusters
        data_slices = split_data_by_clusters(local_data, clusters, scope, rows=True)
        
        if len(data_slices) != 2:
            return -np.inf
        
        left_data_slice, left_scope_slice, left_proportion = data_slices[0]
        right_data_slice, right_scope_slice, right_proportion = data_slices[1]
        
        # Remove conditioning column from analysis
        left_data_reduced = _remove_column(left_data_slice, col_conditioning)
        right_data_reduced = _remove_column(right_data_slice, col_conditioning)
        
        # Compute log-likelihoods
        left_ll = naive_ll(left_data_reduced)
        right_ll = naive_ll(right_data_reduced)
        
        # Compute weighted conditioning log-likelihood
        left_weighted = (left_ll + np.log(left_proportion)) * left_data_reduced.shape[0]
        right_weighted = (right_ll + np.log(right_proportion)) * right_data_reduced.shape[0]
        
        total_samples = left_data_reduced.shape[0] + right_data_reduced.shape[0]
        conditioning_ll = (left_weighted + right_weighted) / total_samples
        
        return conditioning_ll
        
    except Exception as e:
        logger.warning(f"Error evaluating conditioning split for column {col_conditioning}: {e}")
        return -np.inf


def _remove_column(data_array, col_index):
    """Remove a specific column from data array"""
    if col_index < 0 or col_index >= data_array.shape[1]:
        return data_array
    
    if data_array.shape[1] == 1:
        return np.empty((data_array.shape[0], 0))
    
    # Efficiently remove column using concatenation
    if col_index == 0:
        return data_array[:, 1:]
    elif col_index == data_array.shape[1] - 1:
        return data_array[:, :-1]
    else:
        return np.hstack((data_array[:, :col_index], data_array[:, (col_index + 1):]))


def get_split_rows_naive_mle_conditioning():
    """
    Create a function for naive MLE conditioning with improved efficiency
    """
    def split_rows_naive_mle_conditioning(local_data):
        try:
            n_rows, n_cols = _validate_data_for_conditioning(local_data)
            
            if n_cols < 2:
                logger.warning("Need at least 2 columns for MLE conditioning")
                return None, False
            
            scope = list(range(n_cols))
            
            # Find valid conditioning columns
            valid_columns = _find_valid_conditioning_columns(local_data)
            
            if not valid_columns:
                logger.warning("No valid columns for MLE conditioning")
                return None, False
            
            # Evaluate each valid column
            best_col_conditioning = None
            best_conditioning_ll = -np.inf
            
            for col_conditioning in valid_columns:
                conditioning_ll = _evaluate_conditioning_split(local_data, col_conditioning, scope)
                
                if conditioning_ll > best_conditioning_ll:
                    best_conditioning_ll = conditioning_ll
                    best_col_conditioning = col_conditioning
            
            # Return best result
            if best_col_conditioning is not None:
                logger.debug(f"Best conditioning column: {best_col_conditioning} "
                           f"with LL: {best_conditioning_ll}")
                return best_col_conditioning, True
            else:
                logger.warning("No suitable conditioning column found")
                return None, False
            
        except Exception as e:
            logger.error(f"MLE conditioning failed: {e}")
            return None, False
    
    return split_rows_naive_mle_conditioning