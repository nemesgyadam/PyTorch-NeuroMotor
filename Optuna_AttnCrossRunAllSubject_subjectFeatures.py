import os
import mne
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
from src.train import train, plot_metrics

import optuna
from functools import partial
from optuna.samplers import TPESampler

from src.models.attn_cond_eegnet_subjectFeatures import ConditionedEEGNet
from src.dataset.MI_dataset_single_subject import MI_Dataset
from config.over60 import cfg
from config.over60_withbaseline import cfg as cfg_baseline


from utils.eval import accuracy
from utils.model import print_parameters, print_weights_statistics




def extract_features(epochs: mne.Epochs) -> np.ndarray:
    """
    Extracts various features from MNE epochs object across multiple frequency bands,
    and returns them as a numpy array.

    Parameters:
    epochs (mne.Epochs): The epochs object with EEG data.

    Returns:
    np.ndarray: An array containing 12 averaged features across epochs.
    """

    # Define frequency bands
    bands = [(8, 13), (13, 18), (18, 25)]

   
    # Initialize list to collect features
    feature_values = []

    # Extract data from epochs
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    # Calculate features for each band
    for fmin, fmax in bands:
        psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)

        average_power = np.mean(psds, axis=(1, 2))
        sum_power = np.sum(psds, axis=(1, 2))
        peak_frequency = freqs[np.argmax(psds, axis=2)].mean(axis=1)

        feature_values.append(np.mean(average_power))
        feature_values.append(np.mean(sum_power))
        feature_values.append(np.mean(peak_frequency))

    # Overall features across all bands
    psds, _ = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, fmin=8, fmax=25, verbose=False)
    overall_average_power = np.mean(psds)
    overall_sum_power = np.sum(psds)
    overall_std_dev_power = np.std(np.mean(psds, axis=(1, 2)))

    feature_values.extend([overall_average_power, overall_sum_power, overall_std_dev_power])

    return np.array(feature_values)



print("Generating subject features...")
subject_features = {}
for subject_id in cfg_baseline['data']['subjects']:
    print("Subject ID: ", subject_id)
    # TODO exclude test runs
    runs = cfg_baseline['data']['train_runs'][subject_id] + cfg_baseline['data']['test_runs'][subject_id]
    dataset = MI_Dataset(cfg_baseline, subject_id, runs=runs)

    epochs = dataset.epochs
    subject_features[subject_id] = extract_features(epochs)
  

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = MI_Dataset.get_concat_dataset(cfg, split='train', return_subject_number = True, device=device, verbose=False)
train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
print(f"Train dataset: {len(train_dataset)} samples")

test_dataset = MI_Dataset.get_concat_dataset(cfg, split='test', return_subject_number = True, device=device, verbose=False)
test_dataloader = DataLoader(test_dataset, batch_size=cfg['train']['batch_size'], shuffle=False)
print(f"Test dataset: {len(test_dataset)} samples")




# Define the objective function to optimize
def objective(trial, cfg):
    # Sample hyperparameters to optimize
     cfg['train']['n_epochs'] = trial.suggest_int('n_epochs', 1, 15)*10

    # Create and train the model with the sampled hyperparameters
    model = ConditionedEEGNet.from_config(cfg['model'], subject_features, cfg["data"]["subjects"], device)
    loss, val_loss, train_acc, val_acc = train(model, train_dataloader, test_dataloader, cfg['train'])
    
    # Return the validation loss as the objective to minimize

    return float(val_acc[-1]) # val_loss

# Create an Optuna study
study = optuna.create_study(direction='maximize',
     storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
     study_name ="ConditionedEEGNet_subjectFeatures",
     load_if_exists=True
     )

# Define a partial function for the objective function with the cfg parameter
objective_partial = partial(objective, cfg=cfg)

# Optimize the hyperparameters
study.optimize(objective_partial, n_trials=30)

# Get the best hyperparameters
best_params = study.best_params

# Print the best hyperparameters
print("Best Hyperparameters:")
print(best_params)