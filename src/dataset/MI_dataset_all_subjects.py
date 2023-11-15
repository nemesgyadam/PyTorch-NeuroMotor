import os
import numpy as np
import importlib
import mne
from typing import List, Tuple

mne.set_log_level("error")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


MAPPING = {7: "feet", 8: "left_hand", 9: "right_hand", 10: "tongue"}


class MI_Dataset(Dataset):
    def __init__(
        self,
        subject_ids: List[int],
        device: str = "cpu",
        config: str = "default",
        verbose: bool = False,
    ):
        self.data_root = "data"
        self.subject_ids = subject_ids
        self.device = device

        self.load_config(config)

        self.load_raw()
        self.apply_preprocess()
        self.create_epochs()

        self.format_data()

        self.in_timesteps = self.X.shape[-1]
        self.in_channels = self.X.shape[-2]

        if verbose:
            print("#" * 50)
            print("Dataset created:")
            print(f"X --> {self.X.shape} ({self.X.dtype})")
            print(f"y --> {self.y.shape} ({self.y.dtype})")
            print("#" * 50)

    def load_config(self, file: str) -> None:
        cfg = importlib.import_module(f"config.{file}").cfg

        self.target_freq = cfg["preprocessing"]["target_freq"]
        self.low_freq = cfg["preprocessing"]["low_freq"]
        self.high_freq = cfg["preprocessing"]["high_freq"]
        self.average_ref = cfg["preprocessing"]["average_ref"]

        self.baseline = cfg["epochs"]["baseline"]
        self.tmin = cfg["epochs"]["tmin"]
        self.tmax = cfg["epochs"]["tmax"]

        self.normalize = cfg["train"]["normalize"]

    def load_raw(self) -> None:
        self.subject_paths = [
            os.path.join(self.data_root, "A0" + str(subject_id) + "T.gdf")
            for subject_id in self.subject_ids
        ]
        self.raws = [
            mne.io.read_raw_gdf(subject_path, preload=True)
            for subject_path in self.subject_paths
        ]
        for raw in self.raws:
            eog_channels = [
                i for i, ch_name in enumerate(raw.ch_names) if "EOG" in ch_name
            ]
            raw.drop_channels([raw.ch_names[ch_idx] for ch_idx in eog_channels])

        self.filter_events()

    def filter_events(self) -> None:
        for raw in self.raws:
            events, _ = mne.events_from_annotations(raw)
            annot_from_events = mne.annotations_from_events(
                events, event_desc=MAPPING, sfreq=raw.info["sfreq"]
            )
            raw.set_annotations(annot_from_events)

    def apply_preprocess(self) -> None:
        def preprocess_raw(session):
            session = session.resample(self.target_freq, npad="auto")
            if self.average_ref:
                session = session.set_eeg_reference("average", projection=True)
            session = session.filter(l_freq=self.low_freq, h_freq=self.high_freq)
            return session

        self.raws = [preprocess_raw(raw) for raw in self.raws]

    def create_epochs(self):
        def split2epochs(session):
            events, event_ids = mne.events_from_annotations(session)
            return mne.Epochs(
                session,
                events=events,
                event_id=event_ids,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=self.baseline,
                preload=True,
            )

        self.epochs = [split2epochs(raw) for raw in self.raws]
        self.epochs = mne.concatenate_epochs(self.epochs)
        self.epochs = self.epochs.crop(tmin=self.baseline[-1], tmax=self.tmax)
        del self.raws

    def format_data(self):
        self.X = self.epochs.get_data()

        self.y = self.epochs.events[:, -1]
        self.y -= 1  # start at 0

        if self.normalize:
            self.do_normalize()
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def do_normalize(self):
        orig_shape = self.X.shape
        self.X = self.X.reshape(self.X.shape[0], -1)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = self.X.reshape(orig_shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
