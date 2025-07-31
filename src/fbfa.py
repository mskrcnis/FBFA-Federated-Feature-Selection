import numpy as np
from sklearn.feature_selection import f_classif

def compute_fisher_scores(X, y):
    scores, _ = f_classif(X, y)
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val > min_val:
        return (scores - min_val) / (max_val - min_val)
    return np.zeros_like(scores)

def compute_corr_matrix(X):
    return np.abs(np.corrcoef(X, rowvar=False))

def evaluate_feature_subset(subset, fisher_scores, corr_matrix, penalty_lambda=0.7):
    if len(subset) == 0:
        return 0
    fisher_sum = np.sum(fisher_scores[subset])
    if len(subset) > 1:
        corr_penalty = np.sum(corr_matrix[np.ix_(subset, subset)]) - np.sum(np.diag(corr_matrix[subset][:, subset]))
        corr_penalty /= 2
    else:
        corr_penalty = 0.0
    return penalty_lambda * fisher_sum - (1 - penalty_lambda) * corr_penalty

def one_step_binary_firefly(
    firefly_mask_prev, global_mask_prev, local_best_mask_prev,
    fisher_scores, corr_matrix,
    penalty_lambda=0.7, p_global=0.3, p_local=0.3, mutation_rate=0.05
):
    n_features = len(firefly_mask_prev)
    new_mask = firefly_mask_prev.copy()
    for i in range(n_features):
        r = np.random.rand()
        if r < p_global:
            new_mask[i] = global_mask_prev[i]
        elif r < p_global + p_local:
            new_mask[i] = local_best_mask_prev[i]
        elif np.random.rand() < mutation_rate:
            new_mask[i] = 1 - new_mask[i]
    if np.random.rand() < 0.2:
        idx = np.random.randint(n_features)
        new_mask[idx] = 1 - new_mask[idx]
    return new_mask

def run_fbfa_federated(client_data_np, feature_names, n_rounds=20, n_fireflies=20, penalty_lambda=0.8,
                       p_global=0.3, p_local=0.3, mutation_rate=0.05, rho_start=0.2, rho_end=0.8, seed=42):
    np.random.seed(seed)
    num_clients = len(client_data_np)
    n_features = client_data_np[0][0].shape[1]
    client_fisher_scores = []
    client_corr_matrix = []

    for Xc, yc in client_data_np:
        client_fisher_scores.append(compute_fisher_scores(Xc, yc))
        client_corr_matrix.append(compute_corr_matrix(Xc))

    client_fireflies = []
    client_local_bests = []
    for cid in range(num_clients):
        fireflies = []
        best_fitness = -np.inf
        best_mask = None
        for _ in range(n_fireflies):
            mask = np.random.choice([0, 1], size=n_features)
            if np.sum(mask) == 0:
                mask[np.random.randint(n_features)] = 1
            fireflies.append(mask)
            fit = evaluate_feature_subset(np.where(mask)[0], client_fisher_scores[cid], client_corr_matrix[cid], penalty_lambda)
            if fit > best_fitness:
                best_fitness = fit
                best_mask = mask.copy()
        if best_mask is None:
            best_mask = np.ones(n_features, dtype=int)
        client_fireflies.append(fireflies)
        client_local_bests.append(best_mask)

    global_mask = np.ones(n_features, dtype=int)

    for r in range(n_rounds):
        print(f"\n==== Federated BFA Round {r+1} ====")
        rho = rho_start + (rho_end - rho_start) * (r / (n_rounds - 1))
        print(f"  Adaptive rho: {rho:.2f}")
        client_best_masks = []

        for cid in range(num_clients):
            fireflies = client_fireflies[cid]
            fisher_scores = client_fisher_scores[cid]
            corr_matrix = client_corr_matrix[cid]
            local_best = client_local_bests[cid]
            new_fireflies = []
            best_fitness = -np.inf
            best_mask = None

            for f in range(n_fireflies):
                new_mask = one_step_binary_firefly(
                    fireflies[f], global_mask, local_best,
                    fisher_scores, corr_matrix,
                    penalty_lambda, p_global, p_local, mutation_rate
                )
                if np.sum(new_mask) == 0:
                    new_mask[np.random.randint(n_features)] = 1
                new_fireflies.append(new_mask)
                fit = evaluate_feature_subset(np.where(new_mask)[0], fisher_scores, corr_matrix, penalty_lambda)
                if fit > best_fitness:
                    best_fitness = fit
                    best_mask = new_mask.copy()

            if best_mask is None:
                best_mask = np.ones(n_features, dtype=int)
            client_fireflies[cid] = new_fireflies
            client_local_bests[cid] = best_mask.copy()
            client_best_masks.append(best_mask.copy())

        vote_counts = np.sum(client_best_masks, axis=0)
        vote_mask = (vote_counts >= (rho * num_clients)).astype(int)
        print(f"  → Round {r+1} selects {vote_mask.sum()} features")
        global_mask = vote_mask.copy()

    selected_indices = np.where(global_mask == 1)[0]
    print(f"\nFinal selected feature count: {len(selected_indices)}")
    print("Selected feature names:", [feature_names[i] for i in selected_indices])
    return selected_indices

