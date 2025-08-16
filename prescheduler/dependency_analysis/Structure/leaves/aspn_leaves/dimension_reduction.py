from sklearn.decomposition import PCA
import numpy as np
import pickle

def PCA_reduction(data, max_sampling_threshold_rows=100000, threshold=0.9, target_dimensions=None):
    """
    threshold: explain at least how many number of variance
    target_dimensions: specify exact number of dimensions to reduce to (overrides threshold)
    """
    if target_dimensions is not None:
        # 如果指定了目标维度，则直接使用
        k = min(target_dimensions, data.shape[1])
        optimal_pca = PCA(n_components=k)
        if data.shape[0] > max_sampling_threshold_rows:
            data_sample = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]
            optimal_pca.fit(data_sample)
        else:
            optimal_pca.fit(data)
        return optimal_pca, optimal_pca.transform(data)
        
    # 原有的基于阈值的逻辑
    if threshold >= 1:
        return None, data
    if data.shape[0] > max_sampling_threshold_rows:
        data_sample = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]
    else:
        data_sample = data

    temp_pca = PCA(n_components=data.shape[0])
    temp_pca.fit(data_sample)
    explained_var = temp_pca.explained_variance_ratio_
    assert np.isclose(np.sum(explained_var), 1.0), "incorrect PCA"

    total_explain = 0
    for i in range(data.shape[0]):
        total_explain += explained_var[i]
        if total_explain >= threshold:
            k = i+1
            break
    if k == data.shape[0]:
        return None, data
    else:
        optimal_pca = PCA(n_components=k)
        optimal_pca.fit(data_sample)
        return optimal_pca, optimal_pca.transform(data)
