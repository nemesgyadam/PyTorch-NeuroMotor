import os
import numpy as np
import importlib
import mne

mne.set_log_level("error")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

MAPPING = {7: "feet", 8: "left_hand", 9: "right_hand", 10: "tongue"}


class MI_Dataset(Dataset):
    def __init__(self, subject_id, runs, device= "cpu", config= "default", verbose= False):
        self.data_root = "data"
        self.subject_id = subject_id
        self.device = device
        self.runs = runs

        self.load_config(config)

        self.load_raw()
        self.apply_preprocess()
        self.create_epochs()
        if verbose:
            print(self.epochs)

        self.split_by_runs()
        self.format_data()

        if verbose:
            print("#" * 50)
            print("Dataset created:")
            print(f"X --> {self.X.shape} ({self.X.dtype})")
            print(f"y --> {self.y.shape} ({self.y.dtype})")
            print("#" * 50)

    def load_config(self, file):
        cfg = importlib.import_module(f"config.{file}").cfg

        self.target_freq = cfg["preprocessing"]["target_freq"]
        self.low_freq = cfg["preprocessing"]["low_freq"]
        self.high_freq = cfg["preprocessing"]["high_freq"]
        self.average_ref = cfg["preprocessing"]["average_ref"]

        self.baseline = cfg["epochs"]["baseline"]
        self.tmin = cfg["epochs"]["tmin"]
        self.tmax = cfg["epochs"]["tmax"]

        self.normalize = cfg["train"]["normalize"]

    def load_raw(self):
        subject_path = os.path.join(
            self.data_root, "A0" + str(self.subject_id) + "T.gdf"
        )
        self.raw = mne.io.read_raw_gdf(subject_path, preload=True)
        self.filter_events()
        eog_channels = [i for i, ch_name in enumerate(self.raw.ch_names) if 'EOG' in ch_name]
        self.raw.drop_channels([self.raw.ch_names[ch_idx] for ch_idx in eog_channels])


    def filter_events(self):
        events, _ = mne.events_from_annotations(self.raw)
        annot_from_events = mne.annotations_from_events(
            events, event_desc=MAPPING, sfreq=self.raw.info["sfreq"]
        )

        self.raw.set_annotations(annot_from_events)

    def apply_preprocess(self):
        self.raw = self.raw.resample(self.target_freq, npad="auto")
        if self.average_ref:
            self.raw = self.raw.set_eeg_reference("average", projection=True)

        self.raw = self.raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)

    def create_epochs(self):

        events, event_ids = mne.events_from_annotations(self.raw)

        self.epochs = mne.Epochs(
            self.raw,
            events=events,
            event_id=event_ids,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            preload=True,
        )

        self.epochs = self.epochs.crop(tmin=self.baseline[-1], tmax=self.tmax)

        del self.raw

    def split_by_runs(self):
        X = self.epochs.get_data()
    


        if self.normalize:
            orig_shape = X.shape
            X = X.reshape(X.shape[0], -1)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = X.reshape(orig_shape)
            y = self.epochs.events[:, -1]
        y -= 1  # start at 0

        X_by_runs = []
        y_by_runs = []

        #Shuffle
        np.random.seed(42)
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]


        for index in range(0, int(X.shape[0] // 48)):
            X_by_runs.append(X[index * 48 : (index + 1) * 48])
            y_by_runs.append(y[index * 48 : (index + 1) * 48])

        self.runs_features = np.array(X_by_runs)
        self.runs_labels = np.array(y_by_runs)

        self.runs_features = self.runs_features[self.runs]
        self.runs_labels = self.runs_labels[self.runs]

        self.X = self.runs_features.reshape(
            -1, self.runs_features.shape[2], self.runs_features.shape[3]
        )
        self.y = self.runs_labels.reshape(-1)

    def format_data(self):
        # convert to torch tensor
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
