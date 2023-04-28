import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.MI_dataset_single_subject import MI_Dataset
from config.default import cfg
from models.eegnet import EEGNet
from utils.eval import accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_runs = 6


def init_data(subject_id, test_run):
    test_runs = [test_run]
    train_runs = list(range(n_runs))
    train_runs.remove(test_run)

    train_dataset = MI_Dataset(subject_id, train_runs, device=device)
    test_dataset = MI_Dataset(subject_id, test_runs, device=device)

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False
    )

    return train_dataloader, test_dataloader


def init_model(Chans, Samples):
    model = EEGNet(Chans=Chans, Samples=Samples, nb_classes=4)
    model.to(device)
    return model


def train(subject_id, val_run):

    print(f"Training on subject {subject_id} with validation run {val_run}")
    train_dataloader, test_dataloader = init_data(subject_id, val_run)

    model = init_model(
        train_dataloader.dataset.channels, train_dataloader.dataset.time_steps
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    for epoch in range(cfg["train"]["n_epochs"]):
        epoch_loss = 0.0
        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return accuracy(model, test_dataloader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Cross Run validation train on single subject"
    )
    parser.add_argument("subject_id", help="Subject id to train on")
    return parser.parse_args()


def main():
    args = parse_args()
    results = np.array([train(int(args.subject_id), i) for i in range(n_runs)])
    print(f"Mean accuracy: {results.mean():.2f}%")
    print(f"Std accuracy: {results.std():.2f}%")


if __name__ == "__main__":
    main()
