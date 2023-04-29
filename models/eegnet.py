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
        self.dropout2 = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(num_filters2 * (samples // 32), num_classes)

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
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x