import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from src import preprocessing, partitioning, model, train, fbfa, frhc

# ========== Setup ==========
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== Load & Preprocess ==========
X, y, feature_names, label_col = preprocessing.preprocess_csv()
num_classes = len(np.unique(y))

# Dataset wrapper
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

full_dataset = TabularDataset(X, y)
train_idx, test_idx = train_test_split(np.arange(len(full_dataset)), test_size=0.2, stratify=y, random_state=SEED)
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Partition for federated training
num_clients = 5
client_indices = partitioning.partition_tabular_dataset(train_dataset, y, train_idx, num_clients=num_clients, alpha=0.8, seed=SEED)

client_data_np = []
for i in range(num_clients):
    idxs = client_indices[i]
    Xc = X[train_idx][idxs]
    yc = y[train_idx][idxs]
    client_data_np.append((Xc, yc))

# ========== Feature Selection Option ==========
print("\nSelect Feature Selection Method:")
print("1. Full Feature Set")
print("2. FBFA")
print("3. FRHC")
choice = input("Enter choice [1/2/3]: ").strip()

if choice == '2':
    selected_indices = fbfa.run_fbfa_federated(client_data_np, feature_names)
elif choice == '3':
    local_selections = [frhc.frhc_local_feature_selection(Xc) for Xc, _ in client_data_np]
    selected_indices = frhc.frhc_global_intersection(local_selections)
    print(f"\nFRHC selected {len(selected_indices)} features:", [feature_names[i] for i in selected_indices])
else:
    selected_indices = list(range(X.shape[1]))
    print(f"\nUsing all {len(selected_indices)} features")

# ========== Build New Dataset ==========
X_sel = X[:, selected_indices]
input_dim = X_sel.shape[1]
full_dataset = TabularDataset(X_sel, y)
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

client_loaders = []
for i in range(num_clients):
    idxs = client_indices[i]
    client_subset = Subset(train_dataset, idxs)
    client_loader = DataLoader(client_subset, batch_size=128, shuffle=True, drop_last=True)
    client_loaders.append(client_loader)

# ========== Federated Training ==========
global_model = model.TabularMLP(input_dim=input_dim, num_classes=num_classes)
global_model.load_state_dict(global_model.state_dict())

num_rounds = 20
for rnd in range(1, num_rounds + 1):
    adaptive_epochs = max(1, int(10 - 9 * (rnd-1) / (num_rounds-1)))
    print(f"\n[Round {rnd}] Local Epochs: {adaptive_epochs}")
    local_weights = []
    client_acc_before = []
    client_acc_after = []

    for cid in range(num_clients):
        acc_before = train.evaluate_local(global_model, client_loaders[cid])
        local_model = model.TabularMLP(input_dim=input_dim, num_classes=num_classes)
        local_model.load_state_dict(global_model.state_dict())
        local_model = train.train_one_client(local_model, client_loaders[cid], epochs=adaptive_epochs)
        acc_after = train.evaluate_local(local_model, client_loaders[cid])
        local_weights.append(local_model.state_dict())
        client_acc_before.append(acc_before)
        client_acc_after.append(acc_after)
        print(f" Client {cid+1:2d} | Before: {acc_before:5.2f}% | After: {acc_after:5.2f}%")

    global_model.load_state_dict(train.average_weights(local_weights))
    acc_global, cm = train.test_modelv2(global_model, test_loader)
    print(f"\n→ Global Test Accuracy: {acc_global:.2f}%")
    print("Confusion Matrix:\n", cm)

print("\n✅ Training complete.")

