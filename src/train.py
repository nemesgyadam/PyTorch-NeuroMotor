import numpy as np
import random
import torch
import torch.nn as nn

from matplotlib import pyplot as plt

from utils.eval import accuracy

def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_loss(model, dataloader, loss_fnc):
    total_loss = 0
    with torch.no_grad():  # Temporarily turn off gradients for efficiency
        for batch_features, batch_labels in dataloader:
            if isinstance(batch_features, list):
                outputs = model(*batch_features)
            else:
                outputs = model(batch_features)
            loss = loss_fnc(outputs, batch_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)  # average loss

def train(model, train_dataloader, val_dataloader, cfg, verbose=True):
    set_seeds()
    loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    loss_fnc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    for epoch in range(cfg["n_epochs"]):
        epoch_loss = 0.0

        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            if isinstance(batch_features, list):
                outputs = model(*batch_features)
            else:
                outputs = model(batch_features)
            loss = loss_fnc(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)

        loss_history.append(epoch_loss)

        val_loss = calculate_loss(model, val_dataloader, loss_fnc)
        val_loss_history.append(val_loss)

        if epoch % 10 == 9:
            train_accuracy = accuracy(model, train_dataloader)
            val_accuracy = accuracy(model, val_dataloader)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{cfg['n_epochs']}, Loss: {epoch_loss:.5f}, Val Loss: {val_loss:.5f}, Train acc: {train_accuracy:.2f}%, Test acc: {val_accuracy:.2f}%"
                )

    if verbose:
        print("#" * 50)
        print(f"Final train loss: {epoch_loss}")
        print(f"Final val loss: {val_loss}")
        print(f"Final train acc: {accuracy(model, train_dataloader):.2f}%")
        print(f"Final val acc: {accuracy(model, val_dataloader):.2f}%")

    return loss_history, val_loss_history, train_acc_history, val_acc_history


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    # Create lists for the x-axis (epochs)
    epochs_losses = list(range(len(train_losses)))
    epochs_accuracies = list(range(0, len(train_accuracies) * 10, 10))  # We assume accuracies are measured every 10 epochs

    # Create figure with two subplots
    fig, axs = plt.subplots(2, figsize=(10, 12))

    # Plot accuracies
    axs[0].plot(epochs_accuracies, train_accuracies, marker='o', linestyle='-', color='r', label='Train Accuracy')
    axs[0].plot(epochs_accuracies, val_accuracies, marker='o', linestyle='-', color='b', label='Validation Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    # Plot losses
    axs[1].plot(epochs_losses, train_losses, marker='o', linestyle='-', color='r', label='Train Loss')
    axs[1].plot(epochs_losses, val_losses, marker='o', linestyle='-', color='b', label='Validation Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()
