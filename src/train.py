import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_client(model, loader, epochs=1, lr=0.01):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.cpu()

def evaluate_local(model, loader):
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
    acc = 100. * correct / total
    return acc

def test_model(model, loader):
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
    acc = 100. * correct / total
    return acc

def test_modelv2(model, loader):
    model = model.to(device)
    model.eval()
    correct = total = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    acc = 100. * correct / total
    cm = confusion_matrix(all_targets, all_preds)
    return acc, cm

def average_weights(weight_list):
    avg_weights = {}
    for key in weight_list[0].keys():
        avg_weights[key] = sum([w[key] for w in weight_list]) / len(weight_list)
    return avg_weights

