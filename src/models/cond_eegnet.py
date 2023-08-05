import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.models.eegnet import EEGNet
from src.models.conditioned_batch_norm import ConditionedBatchNorm
from src.models.subject_encoder import SubjectEncoder



class ConditionedEEGNet(nn.Module):
    def __init__(
        self,
        classify: bool = True,
        n_classes: int = 4,
        n_subjects: int = 1,
        channels: int = 22,
        time_steps: int = 401,
        kernel_length: int = 64,
        n_filters1: int = 16,
        depth_multiplier: int = 2,
        n_filters2: int = 32,
        dropout_rate: float = 0.5,
        subject_filters: int = 16,
        final_features: int = 4,
        device: str = "cpu",
    ) -> None:

        super(ConditionedEEGNet, self).__init__()
        self.device = device


        ''' Initiate Encoders '''
        self.eeg_encoder = EEGNet(
            classify=False,                     # Remove head
            n_classes=n_classes,
            channels=channels,
            time_steps=time_steps,
            kernel_length=kernel_length,
            n_filters1=n_filters1,
            depth_multiplier=depth_multiplier,
            n_filters2=n_filters2,
            dropout_rate=dropout_rate,

        )
        self.eeg_dim = self.eeg_encoder.calculate_output_dim()

        #self.subject_encoder = SubjectEncoder(n_subjects=n_subjects, n_filters=subject_filters)
        self.subject_encoder = nn.Embedding(n_subjects, subject_filters)
        ''' Initialize Conditioning '''
        self.eeg_bn = ConditionedBatchNorm(self.eeg_dim, n_subjects)
        self.subject_bn = ConditionedBatchNorm(subject_filters, n_subjects)

        self.linear = nn.Linear(subject_filters+self.eeg_dim, final_features)
        self.act = nn.ELU()

        ''' Initialize Classifier '''
        # self.fn1 = nn.Linear(embed_dim, n_filters3)
        # self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_features, n_classes)

        self.to(self.device)

    def forward(self, eeg_data: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        ''' Encoders '''
        eeg_features = self.eeg_encoder(eeg_data)
        eeg_features = self.eeg_bn(eeg_features, subject_id)
        #eeg_features = self.eeg_norm(eeg_features)

        subject_features = self.subject_encoder(subject_id)
        subject_features = self.subject_bn(subject_features, subject_id)
        #subject_features = self.subject_norm(subject_features)

        
        # print(f'eeg_features: {eeg_features.shape}')               #[ B, 384]
        # print(f'subject_features: {subject_features.shape}')    	 #[ B, 16]

        x = torch.cat((eeg_features, subject_features), dim=1)       #[ B, 400]
        x = self.linear(x)
        x = self.act(x)


        ''' Classify '''
        # x = self.fn1(x)
        # x = self.act1(x)
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
