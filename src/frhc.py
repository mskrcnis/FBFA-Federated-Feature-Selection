import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def frhc_local_feature_selection(X, max_clusters=None, comp_feat=1):
    """
    Local representative feature selection by hierarchical clustering of features.
    
    Parameters:
        X: [n_samples, n_features] numpy array (client's local data)
        max_clusters: int or None, max clusters to try for optimal selection
        comp_feat: int, number of compensation features to add

    Returns:
        selected_feature_indices: list of selected feature indices
    """
    n_features = X.shape[1]
    corr_matrix = np.corrcoef(X, rowvar=False)
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)

    Z = linkage(condensed, method='average')
    if max_clusters is None:
        K = int(np.sqrt(n_features))
    else:
        K = min(max_clusters, n_features)
    clusters = fcluster(Z, K, criterion='maxclust')

    cluster_sizes = [(c, np.sum(clusters == c)) for c in np.unique(clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    selected_features = []
    for i in range(min(2, len(cluster_sizes))):
        c = cluster_sizes[i][0]
        selected_features.extend(np.where(clusters == c)[0].tolist())

    if comp_feat > 0:
        for i in range(2, min(2 + comp_feat, len(cluster_sizes))):
            c = cluster_sizes[i][0]
            selected_features.append(np.where(clusters == c)[0][0])

    return sorted(set(selected_features))

def frhc_global_intersection(selected_lists):
    """
    Compute global overlapping federated features as intersection of local sets.
    
    Parameters:
        selected_lists: list of list of feature indices from clients
    Returns:
        final_indices: list of feature indices present in all clients
    """
    final_indices = set(selected_lists[0])
    for feat_set in selected_lists[1:]:
        final_indices &= set(feat_set)
    return sorted(list(final_indices))

