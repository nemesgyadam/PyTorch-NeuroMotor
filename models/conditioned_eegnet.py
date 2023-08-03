import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_classes: int = 4, channels: int = 22, samples: int = 401,
        dropout_rate: float = 0.5, kernel_length: int = 64, num_filters1: int = 16,
        depth_multiplier: int = 2, num_filters2: int = 32, norm_rate: float = 0.25) -> None:
        super(EEGNet, self).__init__()

        self.channels = channels
        self.samples = samples

        # First convolutional block
        # Temporal convolutional to learn frequency filters
        self.conv1 = nn.Conv2d(1, num_filters1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters1)
        
        # Depthwise convolutional block
        # Connected to each feature map individually, to learn frequency-specific spatial filters
        self.dw_conv1 = nn.Conv2d(num_filters1, num_filters1 * depth_multiplier, (channels, 1), groups=num_filters1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters1 * depth_multiplier)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable convolutional block
        # Learns a temporal summary for each feature map individually, 
        # followed by a pointwise convolution, which learns how to optimally mix the feature maps together
        self.sep_conv1 = nn.Conv2d(num_filters1 * depth_multiplier, num_filters1 * depth_multiplier, (1, 16), groups=num_filters1 * depth_multiplier, padding=(0, 8), bias=False)
        self.conv2 = nn.Conv2d(num_filters1 * depth_multiplier, num_filters2, 1, bias=False)
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

class FeedForward(nn.Module):
    def __init__(self, num_subjects: int = 1, num_filters1: int = 16):
        super(FeedForward, self).__init__()
        self.num_subjects = num_subjects
        self.fn1 = nn.Linear(num_subjects, num_filters1)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(x, num_classes=self.num_subjects).to(torch.float32)
        x = self.fn1(x)
        x = self.act(x)
        return x

class ConditionedEEGNet(nn.Module):
    def __init__(self, num_subjects: int = 1, num_classes: int = 4, channels: int = 22, samples: int = 401,
        dropout_rate: float = 0.5, kernel_length: int = 64, num_filters1: int = 16,
        depth_multiplier: int = 2, num_filters2: int = 32, norm_rate: float = 0.25, embed_dim: int = 32, num_filters3: int = 128) -> None:
        super(ConditionedEEGNet, self).__init__()
        self.embed_dim = embed_dim
        #self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=1, batch_first= True, dropout=0.2)

        self.eeg_processor = EEGNet(num_classes=num_classes, channels=channels, samples=samples,
            dropout_rate=dropout_rate, kernel_length=kernel_length, num_filters1=num_filters1,
            depth_multiplier=depth_multiplier, num_filters2=num_filters2, norm_rate=norm_rate)
        self.subject_processor = FeedForward(num_subjects=num_subjects, num_filters1=num_filters1)


        self.query = nn.Linear(384, embed_dim, bias = False)
        self.key = nn.Linear(16, embed_dim, bias = False)
        #self.value = nn.Linear(16, embed_dim, bias = False)
        self.value = nn.Linear(384, embed_dim, bias = False)
        self.fn1 = nn.Linear(embed_dim, num_filters3)
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fn2 = nn.Linear(num_filters3, num_classes)

        self.eeg_fn = nn.Linear(384, embed_dim)


    def forward(self, eeg_data: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        subject_features = self.subject_processor(subject_id)
        eeg_features = self.eeg_processor(eeg_data)
        #print(f'subject_features: {subject_features.shape}')
        #print(f'eeg_features: {eeg_features.shape}')
        #x = self.attention(subject_features, eeg_features, eeg_features)
        Q = self.query(eeg_features)
        K = self.key(subject_features)
        #V = self.value(subject_features)
        V = self.value(eeg_features)

        attn_matrix = Q @ K.transpose(-1, -2)
        normalized_attn_matrix =attn_matrix / self.embed_dim**0.5
        softmaxed_attn_matrix = F.softmax(normalized_attn_matrix, dim=-1)

        x = softmaxed_attn_matrix @ V
        #print(f'x: {x.shape}')
        # Residual path
        eeg_features = self.eeg_fn(eeg_features)
        x = x + eeg_features

        x = self.fn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fn2(x)
        return x