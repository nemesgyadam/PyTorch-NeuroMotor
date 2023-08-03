import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class EEGNet(nn.Module):
    def __init__(self, num_classes: int = 4, channels: int = 22, samples: int = 401,
        dropout_rate: float = 0.5, kernel_length: int = 64, ff_filter: int = 16,
        depth_multiplier: int = 2, num_filters2: int = 32, norm_rate: float = 0.25) -> None:
        super(EEGNet, self).__init__()

        self.channels = channels
        self.samples = samples

        # First convolutional block
        # Temporal convolutional to learn frequency filters
        self.conv1 = nn.Conv2d(1, ff_filter, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(ff_filter)
        
        # Depthwise convolutional block
        # Connected to each feature map individually, to learn frequency-specific spatial filters
        self.dw_conv1 = nn.Conv2d(ff_filter, ff_filter * depth_multiplier, (channels, 1), groups=ff_filter, bias=False)
        self.bn2 = nn.BatchNorm2d(ff_filter * depth_multiplier)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable  convolutional block
        # Learns a temporal summary for each feature map individually, 
        # followed by a pointwise convolution, which learns how to optimally mix the feature maps together
        self.sep_conv1 = nn.Conv2d(ff_filter * depth_multiplier, ff_filter * depth_multiplier, (1, 16), groups=ff_filter * depth_multiplier, padding=(0, 8), bias=False)
        self.conv2 = nn.Conv2d(ff_filter * depth_multiplier, num_filters2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_filters2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        

        # Fully connected layer
        self.flatten = nn.Flatten()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1,  1, self.channels, self.samples)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dw_conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.sep_conv1(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
       

        x = self.flatten(x)

        return x
    
    def calculate_output_dim(self) -> int:
        x = torch.randn(1, 1, self.channels, self.samples)
        x = self(x)
        return x.shape[-1]

class FeedForward(nn.Module):
    def __init__(self, num_subjects: int = 1, ff_filter: int = 16):
        super(FeedForward, self).__init__()
        self.num_subjects = num_subjects
        self.fn1 = nn.Linear(num_subjects, ff_filter)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(x, num_classes=self.num_subjects).to(torch.float32)
        x = self.fn1(x)
        x = self.act(x)
        return x

class ConditionedEEGNet(nn.Module):
    def __init__(self, num_subjects: int = 1, num_classes: int = 4, channels: int = 22, samples: int = 401,
        dropout_rate: float = 0.5, kernel_length: int = 64, ff_filter: int = 16,
        depth_multiplier: int = 2, num_filters2: int = 32, norm_rate: float = 0.25, embed_dim: int = 32, num_filters3: int = 128, init_std: Optional[float] = None) -> None:
        super(ConditionedEEGNet, self).__init__()
        self.embed_dim = embed_dim
        self.init_std = init_std

        self.eeg_processor = EEGNet(num_classes=num_classes, channels=channels, samples=samples,
            dropout_rate=dropout_rate, kernel_length=kernel_length, ff_filter=ff_filter,
            depth_multiplier=depth_multiplier, num_filters2=num_filters2, norm_rate=norm_rate)
        self.eeg_dim = self.eeg_processor.calculate_output_dim()
        self.subject_processor = FeedForward(num_subjects=num_subjects, ff_filter=ff_filter)

        self.subject_norm  = nn.LayerNorm(ff_filter)
        self.eeg_norm = nn.LayerNorm(self.eeg_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.query = nn.Linear(self.eeg_dim, embed_dim, bias = False)    # TODO
        self.key = nn.Linear(ff_filter, embed_dim, bias = False)       
        self.value = nn.Linear(ff_filter, embed_dim, bias = False)
        #self.value = nn.Linear(self.eeg_dim, embed_dim, bias = False)
        self.fn1 = nn.Linear(embed_dim, num_filters3)
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fn2 = nn.Linear(num_filters3, num_classes)

        self.eeg_fn = nn.Linear(self.eeg_dim, embed_dim)

        if init_std is not None:
            self.apply(self._init_weights)

    @classmethod
    def from_config(cls, config):
        return cls(
            num_subjects=config['num_subjects'],
            num_classes=config['n_classes'],
            channels=config['in_chans'],
            samples=config['n_samples'],
            dropout_rate=config['dropout_rate'],
            kernel_length=config['filter_time_length'],
            ff_filter=config['n_filters_time'],
            depth_multiplier=config['depth_multiplier'],
            num_filters2=config['n_filters_spat'],
            norm_rate=config['norm_rate'],
            embed_dim=config['embedding_dim'],
            num_filters3=config['n_filters3'],
            init_std=config['weight_init_std']
        )


    def forward(self, eeg_data: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        subject_features = self.subject_processor(subject_id)
        subject_features = self.subject_norm(subject_features)

        eeg_features = self.eeg_processor(eeg_data)
        eeg_features = self.eeg_norm(eeg_features)


        #print(f'subject_features: {subject_features.shape}')
        #print(f'eeg_features: {eeg_features.shape}')

        Q = self.query(eeg_features)
        K = self.key(subject_features)
        V = self.value(subject_features)
        #V = self.value(eeg_features)

        attn_matrix = Q @ K.transpose(-1, -2)
        normalized_attn_matrix =attn_matrix / self.embed_dim**0.5
        softmaxed_attn_matrix = F.softmax(normalized_attn_matrix, dim=-1)

        x = softmaxed_attn_matrix @ V
        x = self.attn_norm(x)
        #print(f'x: {x.shape}')
        # Residual path
        eeg_features = self.eeg_fn(eeg_features)
        x = x + eeg_features

        x = self.fn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fn2(x)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
       