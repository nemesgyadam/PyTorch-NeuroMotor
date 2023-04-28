import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def accuracy(model, dataloader):
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_predictions

    return accuracy * 100
