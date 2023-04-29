from torch import no_grad
from sklearn.metrics import accuracy_score

def accuracy(model, dataloader):
    all_labels = []
    all_predictions = []

    with no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return accuracy_score(all_labels, all_predictions) * 100
