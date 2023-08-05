import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        classify: bool = True,
        n_classes: int = 4,
        channels: int = 22,
        time_steps: int = 401,

       
        kernel_length: int = 64,
        n_filters1: int = 16,
        depth_multiplier: int = 2,
        n_filters2: int = 32,
        dropout_rate: float = 0.5,

        device: str = "cpu",
    ) -> None:
        super(EEGNet, self).__init__()

        self.channels = channels
        self.time_steps = time_steps
        self.classify = classify

        # First convolutional block
        # Temporal convolutional to learn frequency filters
        self.conv1 = nn.Conv2d(
            1,
            n_filters1,
            (1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(n_filters1)

        # Depthwise convolutional block
        # Connected to each feature map individually, to learn frequency-specific spatial filters
        self.dw_conv1 = nn.Conv2d(
            n_filters1,
            n_filters1 * depth_multiplier,
            (channels, 1),
            groups=n_filters1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(n_filters1 * depth_multiplier)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable convolutional block
        # Learns a temporal summary for each feature map individually,
        # followed by a pointwise convolution, which learns how to optimally mix the feature maps together
        self.sep_conv1 = nn.Conv2d(
            n_filters1 * depth_multiplier,
            n_filters1 * depth_multiplier,
            (1, 16),
            groups=n_filters1 * depth_multiplier,
            padding=(0, 8),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            n_filters1 * depth_multiplier, n_filters2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(n_filters2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.flatten = nn.Flatten()
        if self.classify:
            self.dense = nn.Linear(n_filters2 * (time_steps // 32), n_classes)
        
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.channels, self.time_steps)
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
        if self.classify:
            x = self.dense(x)

        return x

    def calculate_output_dim(self) -> int:
        x = torch.randn(1, 1, self.channels, self.time_steps)
        x = self(x)
        return x.shape[-1]

    @classmethod
    def from_config(cls, config, device='cpu'):
        # Get the signature of the class constructor
        signature = inspect.signature(cls.__init__)

        # Remove the 'self' parameter
        parameters = signature.parameters.copy()
        del parameters['self']

        # Extract the necessary arguments from the config
        kwargs = {name: config[name] for name in parameters.keys() if name in config}
        kwargs['device'] = device
        return cls(**kwargs)