import numpy as np

def partition_tabular_dataset(dataset, labels, train_indices, num_clients=5, alpha=0.8, seed=42):
    np.random.seed(seed)
    targets = np.array(labels)[train_indices]
    num_classes = np.max(targets) + 1
    idxs = np.arange(len(targets))
    client_idx = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = idxs[targets == c]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        split_idxs = np.split(idx_c, proportions)
        for i, idx in enumerate(split_idxs):
            client_idx[i].extend(idx)

    return client_idx

