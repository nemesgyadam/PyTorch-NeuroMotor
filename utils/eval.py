from typing import List
import torch
from torch import no_grad
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def accuracy(model: torch.nn.Module, dataloader: DataLoader) -> float:
    """
    Calculate the accuracy of a model on the given dataloader.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the dataset to evaluate the model on.
    
    Returns:
        float: The accuracy of the model on the dataset, expressed as a percentage.
    """
    all_labels: List[int] = []
    all_predictions: List[int] = []

    with no_grad():
        for features, labels in dataloader:
            if isinstance(features, list):
                outputs = model(*features)
            else:
                outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return accuracy_score(all_labels, all_predictions) * 100
