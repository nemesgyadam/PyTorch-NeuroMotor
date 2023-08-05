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
        embed_dim: int = 8,
        v_from_subject: bool = True,
        residual: bool = False,
        weight_init_std: Optional[float] = None,
        n_filters3: int = 10,
        device: str = "cpu",
    ) -> None:

        super(ConditionedEEGNet, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.v_from_subject = v_from_subject
        self.residual = residual
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

        #self.subject_encoder = SubjectEncoder(n_subjects=n_subjects, n_filters=subject_filters)
        self.subject_encoder = nn.Embedding(n_subjects, subject_filters)
        self.subject_bn = ConditionedBatchNorm(subject_filters, n_subjects)
        self.subject_norm = nn.LayerNorm(subject_filters)
        ''' Initialize Attention '''
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.query = nn.Linear(self.eeg_dim, embed_dim, bias=False)
        self.key = nn.Linear(subject_filters, embed_dim, bias=False)
        if self.v_from_subject:
            self.value = nn.Linear(subject_filters, embed_dim, bias=False)
        else:
            self.value = nn.Linear(self.eeg_dim, embed_dim, bias = False)

        ''' Initialize Residual Path '''
        # Change the dim of EEG features in order to enable residual path 
        if self.residual:
            self.eeg_scaler = nn.Linear(self.eeg_dim, embed_dim)

        ''' Initialize Classifier '''
        self.fn1 = nn.Linear(embed_dim, n_filters3)
        self.act1 = nn.ELU()
        #self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(n_filters3, n_classes)

       

        if weight_init_std is not None:
            self.apply(self._init_weights)

        self.to(self.device)

    def forward(self, eeg_data: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        ''' Encoders '''
        eeg_features = self.eeg_encoder(eeg_data)
        #eeg_features = self.eeg_norm(eeg_features)
        eeg_features = self.eeg_bn(eeg_features, subject_id)

        subject_features = self.subject_encoder(subject_id)
        #subject_features = self.subject_norm(subject_features)
        subject_features = self.subject_bn(subject_features, subject_id)

        # print(f'subject_features: {subject_features.shape}')
        # print(f'eeg_features: {eeg_features.shape}')

        ''' Attention '''
        Q = self.query(eeg_features)
        K = self.key(subject_features)
        if self.v_from_subject:
            V = self.value(subject_features)
        else:
            V = self.value(eeg_features)

        attn_matrix = Q @ K.transpose(-1, -2)
        normalized_attn_matrix = attn_matrix / self.embed_dim**0.5
        softmaxed_attn_matrix = F.softmax(normalized_attn_matrix, dim=-1)
        x = softmaxed_attn_matrix @ V
        x = self.attn_norm(x)

        ''' Residual path '''
        if self.residual:
            eeg_features = self.eeg_fn(eeg_features)
            x = x + eeg_features

        ''' Classify '''
        x = self.fn1(x)
        x = self.act1(x)
        #x = self.dropout(x)
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
