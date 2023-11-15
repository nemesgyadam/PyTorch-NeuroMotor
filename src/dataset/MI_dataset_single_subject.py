import os
import pickle
import numpy as np
import importlib
import mne
from typing import Tuple, List, Union

mne.set_log_level("error")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler

MAPPING = {7: "feet", 8: "left_hand", 9: "right_hand", 10: "tongue"}  #Competion 2a specific


class MI_Dataset(Dataset):
    def __init__(
        self,
        cfg: dict,
        subject_id: int,
        runs: List[int],
        device: Union[str, torch.device] = "cpu",
        return_subject_number: bool = False,
        verbose: bool = False,
    ):
        """
        Initializes MI_Dataset.

        Args:
            subject_id (int): Subject ID to train on.
            runs (List[int]): List of runs to train on.
            device (Union[str, torch.device], optional): Device to use for data. Defaults to "cpu".
            config (str, optional): Configuration file to use. Defaults to "default".
            verbose (bool, optional): If True, print additional information. Defaults to False.
        """
        self.data_root = "data"
        self.subject_id = subject_id
        self.device = device
        self.runs = runs
        self.return_subject_number = return_subject_number
        self.cfg = cfg
        self.parse_config()

        self.load_raw()
        self.apply_preprocess()
        self.create_epochs()

        self.extract_data()
        self.split_by_runs()
        self.format_data()
        self.set_device(self.device)

        self.in_timesteps = self.X.shape[-1]
        self.in_channels = self.X.shape[-2]

        if verbose:
            print("#" * 50)
            print("Dataset created:")
            print(f"X --> {self.X.shape} ({self.X.dtype})")
            print(f"y --> {self.y.shape} ({self.y.dtype})")
            print("#" * 50)

    def parse_config(self) -> None:
        self.target_freq = self.cfg["preprocessing"]["target_freq"]
        self.low_freq = self.cfg["preprocessing"]["low_freq"]
        self.high_freq = self.cfg["preprocessing"]["high_freq"]
        self.average_ref = self.cfg["preprocessing"]["average_ref"]

        self.baseline = self.cfg["epochs"]["baseline"]
        self.tmin = self.cfg["epochs"]["tmin"]
        self.tmax = self.cfg["epochs"]["tmax"]

        self.normalize = self.cfg["train"]["normalize"]

    def load_raw(self) -> None:
        subject_path = os.path.join(
            self.data_root, "A0" + str(self.subject_id) + "T.gdf"
        )
        self.raw = mne.io.read_raw_gdf(subject_path, preload=True)
        self.filter_events()
        eog_in_channels = [
            i for i, ch_name in enumerate(self.raw.ch_names) if "EOG" in ch_name
        ]
        self.raw.drop_channels([self.raw.ch_names[ch_idx] for ch_idx in eog_in_channels])

    def filter_events(self) -> None:
        events, _ = mne.events_from_annotations(self.raw)
        annot_from_events = mne.annotations_from_events(
            events, event_desc=MAPPING, sfreq=self.raw.info["sfreq"]
        )

        self.raw.set_annotations(annot_from_events)

    def apply_preprocess(self) -> None:
        if self.target_freq:
            self.raw = self.raw.resample(self.target_freq, npad="auto")
        if self.average_ref:
            self.raw = self.raw.set_eeg_reference("average", projection=True)

        self.raw = self.raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)

    def create_epochs(self) -> None:
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
        tmin = self.tmin
        if self.baseline:
            tmin = self.baseline[-1]
        self.epochs = self.epochs.crop(tmin=tmin, tmax=self.tmax)

        del self.raw

    def extract_data(self) -> None:
        self.X = self.epochs.get_data()

        if self.normalize:
            self.do_normalize()

        self.y = self.epochs.events[:, -1]
        self.y -= 1  # start at 0

    def do_normalize(self) -> None:
        orig_shape = self.X.shape
        self.X = self.X.reshape(self.X.shape[0], -1)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = self.X.reshape(orig_shape)

    def split_by_runs(self) -> None:    #Competion 2a specific
        X_by_runs = []
        y_by_runs = []

        for index in range(0, int(self.X.shape[0] // 48)):
            X_by_runs.append(self.X[index * 48 : (index + 1) * 48])
            y_by_runs.append(self.y[index * 48 : (index + 1) * 48])

        self.runs_features = np.array(X_by_runs)
        self.runs_labels = np.array(y_by_runs)

        self.X = self.runs_features[self.runs]
        self.y = self.runs_labels[self.runs]

    def format_data(self) -> None:
        # Remove Run dimension
        self.X = self.X.reshape(-1, self.X.shape[2], self.X.shape[3])
        self.y = self.y.reshape(-1)

        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()

    def set_device(self, device):
        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.return_subject_number:
            subject_number = self.cfg['data']['subjects'].index(self.subject_id)
            return ((self.X[idx], torch.tensor(subject_number, dtype=torch.int64, device = self.device)), self.y[idx])
        else:
         return (self.X[idx],  self.y[idx])

    
    def get_concat_dataset(cfg, split='train', return_subject_number = False, device = None, verbose = False):
        cache_root = 'cache'
        if return_subject_number:
            cache_type = 'all_subjects_with_id'
        else:
            cache_type = 'all_subjects'
        if cfg['data']['subjects'] != list(range(1,10)):
            subject_str =  "-".join(str(s) for s in cfg['data']['subjects'])
            cache_type = cache_type  + '_' + subject_str
            
        def create_dataset(cfg, split='train', return_subject_number = False, device=None, verbose=False):
            return ConcatDataset([
                MI_Dataset(cfg, subject,cfg['data'][f'{split}_runs'][subject], 
                                        return_subject_number=return_subject_number, device=device, verbose=verbose) 
                for subject in cfg['data']['subjects']
            ])

        path = os.path.join(cache_root, cache_type, f'{split}_dataset.pkl')
        if os.path.isfile(path):
            print(f'Loading dataset from {path}...')
            dataset = pickle.load(open(path, 'rb'))
        else:
            print('Creating dataset...')
            dataset = create_dataset(cfg, split=split, return_subject_number = return_subject_number, device=device, verbose=False)
            for d in dataset.datasets:
                d.set_device(device)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f'Saving dataset to {path}...')
            pickle.dump(dataset, open(path, 'wb'))
        return dataset