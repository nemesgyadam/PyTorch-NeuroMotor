import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.models.eegnet import EEGNet
from src.models.utils.conditioned_batch_norm import ConditionedBatchNorm
from src.models.utils.subject_encoder import SubjectEncoder


class ConditionedEEGNet(nn.Module):
    def __init__(
        self,
        subject_averages: dict = {},
        subjects: list = [],
        # EEGEncoder params
        n_subjects: int = 1,
        n_classes: int = 4,
        in_channels: int = 22,
        in_timesteps: int = 401,
        n_time_filters: int = 16,
        time_filter_length: int = 64,
        n_depth_filters: int = 32,
        n_sep_filters: int = 32,
        dropout_rate: float = 0.5,
        weight_init_std: Optional[float] = None,
        # SubjectEncoder params
        n_time_filters_subject: int = 2,
        n_depth_filters_subject: int = 2,
        n_sep_filters_subject: int = 2,
        # Conditioning params
        embed_dim: int = 16,
        device: str = "cpu",
    ) -> None:
        super(ConditionedEEGNet, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.subjects = subjects

        self.weight_init_std = weight_init_std
        self.subject_averages = subject_averages

        """ EEG Encoder """
        self.eeg_encoder = EEGNet(
            classify=False,  # Remove head
            n_classes=n_classes,
            in_channels=in_channels,
            in_timesteps=in_timesteps,
            n_time_filters=n_time_filters,
            time_filter_length=time_filter_length,
            n_depth_filters=n_depth_filters,
            n_sep_filters=n_sep_filters,
            dropout_rate=dropout_rate,
            weight_init_std=weight_init_std,
        )
        self.eeg_dim = self.eeg_encoder.calculate_output_dim()
        self.eeg_bn = ConditionedBatchNorm(self.eeg_dim, n_subjects)
        self.eeg_norm = nn.LayerNorm(self.eeg_dim)
        self.eeg_dim_reduction = nn.Linear(self.eeg_dim, self.embed_dim)

        """ Subject Encoder """
        self.subject_encoder = EEGNet(
            classify=False,  # Remove head
            n_classes=n_classes,
            in_channels=in_channels,
            in_timesteps=in_timesteps,
            n_time_filters=n_time_filters_subject,
            time_filter_length=time_filter_length,
            n_depth_filters=n_depth_filters_subject,
            n_sep_filters=n_sep_filters_subject,
            dropout_rate=dropout_rate,
            weight_init_std=weight_init_std,
        )
        self.subject_dim = self.subject_encoder.calculate_output_dim()

        self.subject_bn = ConditionedBatchNorm(self.subject_dim, n_subjects)
        self.subject_norm = nn.LayerNorm(self.subject_dim)

        self.subject_dim_reduction = nn.Linear(self.subject_dim, self.embed_dim)
        self.act1 = nn.ELU()

        """ Initialize Classifier """
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.embed_dim, n_classes)

        if weight_init_std is not None:
            self.apply(self._init_weights)

        self.to(self.device)

    def forward(
        self, eeg_data: torch.Tensor, subject_numbers: torch.Tensor
    ) -> torch.Tensor:
        """EEG Encoder"""
        eeg_features = self.eeg_encoder(eeg_data)
        eeg_features = self.eeg_norm(eeg_features)
        # eeg_features = self.eeg_bn(eeg_features, subject_numbers)
        eeg_features = self.eeg_dim_reduction(eeg_features)

        """ Subject Encoder """
        ###########   ID2AVG  ################
        subject_averages = []
        for subject_number in subject_numbers:
            subject_id = self.subjects[subject_number.item()]
            subject_averages.append(torch.from_numpy(self.subject_averages[subject_id]))
        subject_average = torch.stack(subject_averages)
        subject_average = subject_average.to(self.device)
        ######################################
        subject_features = self.subject_encoder(subject_average)
        subject_features = self.subject_norm(subject_features)
        # subject_features = self.subject_bn(subject_features, subject_numbers)
        subject_features = self.subject_dim_reduction(subject_features)

        """ Attention """
        subject_features = nn.functional.sigmoid(subject_features)
        x = torch.mul(subject_features, eeg_features)
        x = self.act1(x)

        """ Classify """
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.weight_init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config, subject_averages, subjects, device="cpu"):
        # Get the signature of the class constructor
        signature = inspect.signature(cls.__init__)

        # Remove the 'self' parameter
        parameters = signature.parameters.copy()
        del parameters["self"]

        # Extract the necessary arguments from the config
        kwargs = {name: config[name] for name in parameters.keys() if name in config}
        kwargs["device"] = device
        kwargs["subjects"] = subjects
        kwargs["subject_averages"] = subject_averages

        return cls(**kwargs)
