import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class EEGNet(nn.Module):
    def __init__(
        self,
        classify: bool = True,
        n_classes: int = 4,
        in_channels: int = 22,
        in_timesteps: int = 401,
        n_time_filters: int = 16,
        time_filter_length: int = 64,
        depth_multiplier: int = 2,
        n_sep_filters: int = 32,
        dropout_rate: float = 0.5,
        weight_init_std: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        super(EEGNet, self).__init__()

        self.in_channels = in_channels
        self.in_timesteps = in_timesteps
        self.classify = classify

        n_depth_filters = n_time_filters * depth_multiplier

        self.weight_init_std = weight_init_std

        # Temporal convolutional to learn frequency filters
        self.conv1 = nn.Conv2d(
            1,
            n_time_filters,
            (1, time_filter_length),
            padding=(0, time_filter_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(n_time_filters)

        # Depthwise convolutional block
        # Connected to each feature map individually, to learn frequency-specific spatial filters
        self.dw_conv1 = nn.Conv2d(
            n_time_filters,
            n_depth_filters,
            (in_channels, 1),
            groups=n_time_filters,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(n_depth_filters)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable convolutional block
        # Learns a temporal summary for each feature map individually,
        # followed by a pointwise convolution, which learns how to optimally mix the feature maps together
        self.sep_conv1 = nn.Conv2d(
            n_depth_filters,
            n_depth_filters,
            (1, 16),
            groups=n_depth_filters,
            padding=(0, 8),
            bias=False,
        )
        self.conv2 = nn.Conv2d(n_depth_filters, n_sep_filters, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_sep_filters)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.flatten = nn.Flatten()
        if self.classify:
            self.dense = nn.Linear(n_sep_filters * (in_timesteps // 32), n_classes)

        if weight_init_std is not None:
            self.apply(self._init_weights)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.in_channels, self.in_timesteps)
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
        x = torch.randn(1, 1, self.in_channels, self.in_timesteps)
        x = self(x)
        return x.shape[-1]

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
