import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.train import train, plot_metrics

import optuna
from functools import partial


from config.over60 import cfg
from src.dataset.MI_dataset_single_subject import MI_Dataset

from src.models.eegnet import EEGNet
from src.models.cat_cond_eegnet import ConditionedEEGNet as CatConditionedEEGNet
from src.models.attn_cond_eegnet import ConditionedEEGNet as AttnConditionedEEGNet
from src.models.attn_cond_eegnet_subjectAverages import (
    ConditionedEEGNet as AttnConditionedEEGNet_subjectAverages,
)
from src.models.attn_cond_eegnet_subjectFeatures import (
    ConditionedEEGNet as AttnConditionedEEGNet_subjectFeatures,
)


from utils.extract_epoch_features import extract_epoch_features

METHODS = ["baseline", "cat", "attn", "attn_subjectAverages", "attn_subjectFeatures"]


def parse_args(args):
    parser = argparse.ArgumentParser(description="Bayesn hypertune.")
    parser.add_argument(
        "method",
        type=str,
        help="baseline OR cat OR attn OR attn_subjectAverages OR attn_subjectFeatures",
    )
    parser.add_argument("--n_runs", type=int, default = 2, help="Number of runs to hypertune.")
    args = parser.parse_args(args)
    assert args.method in METHODS, "Invalid method"
    return args


def set_seeds(seed: int = 42) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


def generate_subject_features() -> dict:
    print("Generating subject features...")
    subject_features = {}
    for subject_id in cfg_baseline["data"]["subjects"]:
        print("Subject ID: ", subject_id)
        # TODO exclude test runs
        runs = (
            cfg_baseline["data"]["train_runs"][subject_id]
            + cfg_baseline["data"]["test_runs"][subject_id]
        )
        dataset = MI_Dataset(cfg_baseline, subject_id, runs=runs)

        epochs = dataset.epochs
        subject_features[subject_id] = extract_epoch_features(epochs)
    return subject_features


def generate_subject_averages() -> dict:
    print("Generating subject averages...")
    subject_averages = {}
    for subject_id in cfg["data"]["subjects"]:
        print("Subject ID: ", subject_id)
        # TODO exclude test runs
        runs = (
            cfg["data"]["train_runs"][subject_id] + cfg["data"]["test_runs"][subject_id]
        )
        dataset = MI_Dataset(cfg, subject_id, runs=runs)
        subject_average = np.average(dataset.X, axis=0)
        subject_averages[subject_id] = subject_average
    return subject_averages


def objective(trial, method, cfg, train_dataloader, test_dataloader):
    set_seeds()

    cfg["model"]["n_time_filters"] = trial.suggest_int("n_time_filters", 8, 32)
    #cfg["model"]["time_filter_length"] = trial.suggest_int("time_filter_length", 32, 128)
    cfg["model"]["depth_multiplier"] = trial.suggest_int("n_depth_filters", 1,4)
    cfg["model"]["n_sep_filters"] = trial.suggest_int("n_sep_filters", 8, 32)
    cfg["model"]["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.5)
    cfg["model"]["weight_init_std"] = trial.suggest_float("weight_init_std", 0.0, 0.5)
    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, log=True
    )
    cfg["train"]["epochs_perten"] = trial.suggest_int("n_epochs", 1, 15) * 10
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    )

    # Create model
    if method == "baseline":
        model = EEGNet.from_config(cfg["model"], device=device)
    elif method == "cat":
        model = CatConditionedEEGNet.from_config(cfg["model"], device=device)
    elif method == "attn":
        model = AttnConditionedEEGNet.from_config(cfg["model"], device=device)
    elif method == "attn_subjectAverages":
        model = AttnConditionedEEGNet_subjectAverages.from_config(
            cfg["model"], subject_averages, cfg["data"]["subjects"], device
        )
    elif method == "attn_subjectFeatures":
        model = AttnConditionedEEGNet_subjectFeatures.from_config(
            cfg["model"], subject_features, cfg["data"]["subjects"], device
        )

    # Train model
    val_acc = train(
        model, train_dataloader, test_dataloader, cfg["train"], return_hist=False
    )

    return float(val_acc)


def main(args=None):
    args = parse_args(args)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataloaders
    train_dataset = MI_Dataset.get_concat_dataset(
        cfg, split="train", return_subject_number=args.method!='baseline', device=device, verbose=False
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    print(f"Train dataset: {len(train_dataset)} samples")

    test_dataset = MI_Dataset.get_concat_dataset(
        cfg, split="test", return_subject_number=args.method!='baseline', device=device, verbose=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False
    )
    print(f"Test dataset: {len(test_dataset)} samples")


    # Generate additional features if neccecary
    if args.method == "attn_subjectAverages":
        global subject_averages
        subject_averages = generate_subject_averages()
    elif args.method == "attn_subjectFeatures":
        global subject_features
        subject_features = generate_subject_features()


    print("Starting hypertune! Method: ", args.method)
    input("Press Enter to continue...")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3", 
        study_name=f"over60_{args.method}",
        load_if_exists=True,
    )
    objective_partial = partial(objective, method=args.method, cfg=cfg, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    study.optimize(objective_partial, n_trials=args.n_runs)

    print()
    print("Best Hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()
