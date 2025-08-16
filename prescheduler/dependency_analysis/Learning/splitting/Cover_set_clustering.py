import numpy as np
import logging
import math
from Calculate.inference import EPSILON

logger = logging.getLogger(__name__)


def get_optimal_attribute(rdc_op):
    """
    Select optimal attribute to split on using pairwise RDC matrix
    """
    rdc_mat, scope_loc, condition_loc = rdc_op
    
    if not condition_loc or not scope_loc:
        raise ValueError("Empty condition or scope locations")
    
    # Vectorized computation of correlation scores
    corr_scores = np.zeros(len(condition_loc))
    
    for i, c in enumerate(condition_loc):
        # Get maximum RDC value for this condition across all scope variables
        rdc_values = [rdc_mat[c][s] for s in scope_loc]
        corr_scores[i] = np.max(rdc_values)
    
    # Find attribute with minimum correlation
    opt_idx = np.argmin(corr_scores)
    opt_attr = condition_loc[opt_idx]
    
    logger.info(f"Optimal attribute found at index {opt_idx} with score {corr_scores[opt_idx]}")
    return opt_attr, opt_idx


def _validate_score_matrix(score):
    """Validate score matrix dimensions and values"""
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("Score matrix must be square")
    
    if score.shape[0] == 0:
        raise ValueError("Score matrix cannot be empty")
    
    return score.shape[0]


def _compute_level_range(level, dim):
    """Compute the range for a given level in the hierarchy"""
    step_size = pow(2, level)
    max_nodes = dim // step_size
    return step_size, max_nodes


def _update_division_point(score, mask, pos_start, pos_end, pos_mid):
    """Update division point in the mask"""
    mask[pos_start:pos_end] = 0
    mask[pos_end] = 1
    
    logger.debug(f"Division point updated: [{pos_start}:{pos_end}] -> "
                 f"[{pos_start}:{pos_mid}] + [{pos_mid}:{pos_end}]")


def _update_score_matrix(score, pos_start, pos_end, pos_mid):
    """Update score matrix with computed values"""
    score[pos_start, pos_end] = score[pos_start, pos_mid] + score[pos_mid, pos_end]
    
    logger.debug(f"Score updated: [{pos_start}:{pos_end}] -> "
                 f"[{pos_start}:{pos_mid}] + [{pos_mid}:{pos_end}]")


def find_Cover_Set(score):
    """
    Find cover set using dynamic programming approach with improved efficiency
    """
    dim = _validate_score_matrix(score)
    
    # Initialize mask
    mask = np.ones(dim + 1)
    mask[0] = 0
    
    # Compute hierarchy height
    height = math.ceil(math.log2(dim)) if dim > 1 else 1
    
    # Bottom-up scanning through levels
    for level in range(1, height + 1):
        step_size, max_nodes = _compute_level_range(level, dim)
        
        # Level-wise scanning through nodes
        for node_idx in range(max_nodes + 1):
            pos_start = node_idx * step_size + 1
            pos_next = (node_idx + 1) * step_size
            
            # Check if this node is valid
            if pos_next >= dim and pos_start >= dim:
                continue
            
            pos_end = min(pos_next, dim - 1)
            
            if pos_start >= pos_end:
                continue
            
            pos_mid = math.ceil((pos_end + pos_start) / 2)
            
            # Ensure valid indices
            if pos_mid >= dim or pos_start >= dim or pos_end >= dim:
                continue
            
            try:
                current_score = score[pos_start, pos_end]
                split_score = score[pos_start, pos_mid] + score[pos_mid, pos_end]
                
                if current_score < split_score:
                    _update_division_point(score, mask, pos_start, pos_end, pos_mid)
                else:
                    _update_score_matrix(score, pos_start, pos_end, pos_mid)
                    
            except IndexError as e:
                logger.warning(f"Index error at positions [{pos_start}:{pos_end}]: {e}")
                continue
    
    # Extract cover set
    cover_indices = np.where(mask == 1)[0]
    logger.debug(f"Cover set found: {cover_indices}")
    
    return cover_indices


def _compute_entropy(probabilities):
    """Compute entropy from probability array"""
    # Filter out zero probabilities to avoid log(0)
    nonzero_probs = probabilities[probabilities > 0]
    
    if len(nonzero_probs) == 0:
        return 0.0
    
    return -np.sum(nonzero_probs * np.log2(nonzero_probs))


def get_entropy_matrix(data):
    """
    Create entropy matrix H where H[i,j] records entropy from part i to part j
    """
    if len(data) == 0:
        return np.zeros((1, 1))
    
    n = len(data)
    categories = sorted(np.unique(data))
    k = len(categories)
    
    # Create mapping from category to index
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Compute probability for each category
    probs = np.array([np.sum(data == cat) / n for cat in categories])
    
    # Validate probabilities
    if not np.isclose(np.sum(probs), 1.0):
        logger.warning(f"Probabilities don't sum to 1: {np.sum(probs)}")
        probs = probs / np.sum(probs)  # Normalize
    
    # Initialize entropy matrix
    H = np.zeros((k + 1, k + 1))
    
    # Compute entropy for all intervals
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            if i == j:
                H[i, j] = 0.0
            else:
                interval_probs = probs[i:j]
                if np.sum(interval_probs) > 0:
                    normalized_probs = interval_probs / np.sum(interval_probs)
                    H[i, j] = _compute_entropy(normalized_probs)
                else:
                    H[i, j] = 0.0
    
    return H


def _create_clusters_from_cover_set(data, cover_indices, attr):
    """Create clusters based on cover set indices"""
    n_data = len(data)
    clusters = np.zeros(n_data)
    rect_range = []
    
    if len(cover_indices) == 0:
        # Fallback: single cluster
        rect_range.append({attr: [(np.min(data), np.max(data) + EPSILON)]})
        return clusters, rect_range
    
    categories = sorted(np.unique(data))
    
    # Create clusters based on cover set
    cluster_id = 0
    prev_idx = 0
    
    for cover_idx in cover_indices:
        if cover_idx > len(categories):
            continue
        
        # Define cluster range
        if prev_idx < len(categories) and cover_idx <= len(categories):
            start_val = categories[prev_idx] if prev_idx < len(categories) else categories[-1]
            end_val = categories[min(cover_idx - 1, len(categories) - 1)]
            
            # Assign data points to cluster
            mask = (data >= start_val) & (data <= end_val)
            clusters[mask] = cluster_id
            
            # Add to range list
            rect_range.append({attr: [(start_val, end_val + EPSILON)]})
            
            cluster_id += 1
            prev_idx = cover_idx
    
    return clusters, rect_range


def get_optimal_split_CS(data, attr, n_clusters=2, sample_size=1000000):
    """
    Split attribute using Cover Set approach with improved efficiency
    """
    if len(data) == 0:
        return np.array([]), []
    
    clusters = np.zeros(len(data))
    unique_vals = np.unique(data)
    
    # Handle simple cases
    if len(unique_vals) <= n_clusters:
        rect_range = []
        for i, val in enumerate(unique_vals):
            mask = data == val
            clusters[mask] = i
            rect_range.append({attr: [(val, val + EPSILON)]})
        return clusters, rect_range
    
    try:
        # Sample data if necessary
        if len(data) > sample_size:
            sample_indices = np.random.randint(0, len(data), size=sample_size)
            data_sample = data[sample_indices]
        else:
            data_sample = data
        
        # Get entropy matrix
        H = get_entropy_matrix(data_sample)
        
        # Find cover set
        cover_indices = find_Cover_Set(H)
        
        # Create clusters from cover set
        clusters, rect_range = _create_clusters_from_cover_set(data, cover_indices, attr)
        
        logger.info(f"Cover set clustering completed with {len(rect_range)} clusters")
        return clusters, rect_range
        
    except Exception as e:
        logger.error(f"Cover set clustering failed: {e}")
        
        # Fallback to simple splitting
        if n_clusters == 2:
            median_val = np.median(data)
            clusters[data <= median_val] = 0
            clusters[data > median_val] = 1
            
            rect_range = [
                {attr: [(np.min(data), median_val)]},
                {attr: [(median_val + EPSILON, np.max(data) + EPSILON)]}
            ]
        else:
            # Equal-sized clusters
            sorted_indices = np.argsort(data)
            cluster_size = len(data) // n_clusters
            
            rect_range = []
            for i in range(n_clusters):
                start_idx = i * cluster_size
                end_idx = (i + 1) * cluster_size if i < n_clusters - 1 else len(data)
                
                cluster_indices = sorted_indices[start_idx:end_idx]
                clusters[cluster_indices] = i
                
                start_val = data[cluster_indices[0]]
                end_val = data[cluster_indices[-1]]
                rect_range.append({attr: [(start_val, end_val + EPSILON)]})
        
        return clusters, rect_range


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    dim = 8
    H = np.random.randint(0, 100, (dim + 1, dim + 1))
    H[5, 5] = 0
    H[6, 6] = 0
    H[5, 7] = 888
    H[1, 8] = 999
    
    logger.info("Testing Cover Set algorithm")
    logger.info(f"Input matrix:\n{H[1:, 1:]}")
    
    try:
        cover_set = find_Cover_Set(H)
        logger.info(f"Cover set result: {cover_set}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
