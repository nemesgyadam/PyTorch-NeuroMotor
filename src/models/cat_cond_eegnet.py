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
        subject_filters: int = 16,
        # Dense params
        final_features: int = 4,
        device: str = "cpu",
    ) -> None:

        super(ConditionedEEGNet, self).__init__()
        self.device = device


        ''' EEG Encoder '''
        self.eeg_encoder = EEGNet(
            classify=False,                     # Remove head
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

        ''' Subject Encoder '''
        self.subject_encoder = nn.Embedding(n_subjects, subject_filters)
        self.subject_bn = ConditionedBatchNorm(subject_filters, n_subjects)

        ''' Conditioning '''
        self.linear = nn.Linear(subject_filters+self.eeg_dim, final_features)
        self.act = nn.ELU()

        ''' Initialize Classifier '''
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_features, n_classes)

        self.to(self.device)

    def forward(self, eeg_data: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        ''' Encoders '''
        eeg_features = self.eeg_encoder(eeg_data)
        eeg_features = self.eeg_bn(eeg_features, subject_id)

        subject_features = self.subject_encoder(subject_id)
        subject_features = self.subject_bn(subject_features, subject_id)

        ''' Conditioning '''
        x = torch.cat((eeg_features, subject_features), dim=1)       #[ B, 400]
        x = self.linear(x)
        x = self.act(x)

        ''' Classify '''
        x = self.dropout(x)
        x = self.classifier(x)
        return x



    @classmethod
    def from_config(cls, config, device="cpu"):
        # Get the signature of the class constructor
        signature = inspect.signature(cls.__init__)

        # Remove the 'self' parameter
        parameters = signature.parameters.copy()
        del parameters["self"]

        # Extract the necessary arguments from the config
        kwargs = {name: config[name] for name in parameters.keys() if name in config}
        kwargs["device"] = device
        
        return cls(**kwargs)
