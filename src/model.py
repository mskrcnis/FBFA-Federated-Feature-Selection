import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        features = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(features)
        out = self.fc3(x)

        if return_features:
            return out, features
        else:
            return out

