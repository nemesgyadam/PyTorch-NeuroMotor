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
        embed_dim: int = 16,
        weight_init_std: Optional[float] = None,

        device: str = "cpu",
    ) -> None:

        super(ConditionedEEGNet, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

     
        self.weight_init_std = weight_init_std


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
        self.eeg_bn = ConditionedBatchNorm(self.eeg_dim, n_subjects)
        self.eeg_norm = nn.LayerNorm(self.eeg_dim)
        self.eeg_dim_reduction = nn.Linear(self.eeg_dim, self.embed_dim)

        self.subject_encoder = SubjectEncoder(n_subjects=n_subjects, n_filters=self.embed_dim)
        #self.subject_encoder = nn.Embedding(n_subjects, self.embed_dim)
        self.subject_bn = ConditionedBatchNorm(self.embed_dim, n_subjects)
        self.subject_norm = nn.LayerNorm(self.embed_dim)

    
        self.act1 = nn.ELU()
    

        ''' Initialize Classifier '''
  
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.embed_dim, n_classes)

       

        if weight_init_std is not None:
            self.apply(self._init_weights)

        self.to(self.device)

    def forward(self, eeg_data: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        ''' Encoders '''
        eeg_features = self.eeg_encoder(eeg_data)
        #eeg_features = self.eeg_norm(eeg_features)
        eeg_features = self.eeg_bn(eeg_features, subject_id)
        eeg_features = self.eeg_dim_reduction(eeg_features)

        subject_features = self.subject_encoder(subject_id)
        #subject_features = self.subject_norm(subject_features)
        subject_features = self.subject_bn(subject_features, subject_id)

        #print(f'subject_features: {subject_features.shape}')
        #print(f'eeg_features: {eeg_features.shape}')

        ''' Attention '''
        subject_features = nn.functional.sigmoid(subject_features)
        x = torch.mul(subject_features, eeg_features)
        x = self.act1(x)
      

        ''' Classify '''
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.weight_init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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
