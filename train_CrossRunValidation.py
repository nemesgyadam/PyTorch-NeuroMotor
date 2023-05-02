import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.MI_dataset_single_subject import MI_Dataset
import importlib
from models.eegnet import EEGNet
from utils.eval import accuracy
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_runs = 6


def init_data(subject_id: int, test_run: int) -> Tuple[DataLoader, DataLoader]:
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


def init_model(channels: int, samples: int) -> torch.nn.Module:
    model = EEGNet(channels=channels, samples=samples, num_classes=4)
    model.to(device)
    return model


def train(subject_id: int, val_run: int) -> float:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cross Run validation train on single subject"
    )
    parser.add_argument("subject_id", help="Subject id to train on")
    parser.add_argument(
        "--config", default="default", help="Config file placed in config/ directory"
    )
    global cfg
    try:
        cfg = importlib.import_module(f"config.{parser.parse_args().config}").cfg
    except ModuleNotFoundError:
        print(
            f"Config file {parser.parse_args().config} not found in config/ directory"
        )
        exit(1)

    print(f"Loaded config file: {parser.parse_args().config}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = np.array([train(int(args.subject_id), i) for i in range(n_runs)])
    print(f"Mean accuracy: {results.mean():.2f}%")
    print(f"Std accuracy: {results.std():.2f}%")


if __name__ == "__main__":
    main()
