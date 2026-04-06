"""Temporal Convolutional Network (TCN).

Implements the architecture described in Bai et al. (2018), "An Empirical
Evaluation of Generic Convolutional and Recurrent Networks for Sequence
Modeling", with weight-normalised dilated causal convolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TemporalBlock(nn.Module):
    """Single TCN block: two dilated causal convolutions with a residual path.

    Each convolution is followed by weight normalisation, ReLU, and dropout.
    A 1×1 convolution is used on the residual path when the channel count
    changes.

    Args:
        n_inputs: Number of input channels.
        n_outputs: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Dilation factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
            ),
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
            ),
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual projection when channel dimensions differ
        self.downsample: nn.Module = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        )
        self.relu_out = nn.ReLU()
        self._padding = padding
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise convolution weights with Kaiming normal."""
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if isinstance(self.downsample, nn.Conv1d):
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the temporal block.

        Args:
            x: Input tensor of shape ``[batch, n_inputs, time]``.

        Returns:
            Output tensor of shape ``[batch, n_outputs, time]``.
        """
        # First causal conv + activation + dropout
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]          # trim future padding (causal)
        out = self.relu1(out)                 # [batch, n_outputs, time]
        out = self.dropout1(out)

        # Second causal conv + activation + dropout
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]          # trim future padding (causal)
        out = self.relu2(out)                 # [batch, n_outputs, time]
        out = self.dropout2(out)

        # Residual connection
        res = self.downsample(x)              # [batch, n_outputs, time]
        return self.relu_out(out + res)       # [batch, n_outputs, time]


class TemporalConvNet(nn.Module):
    """Stack of :class:`TemporalBlock` layers with exponentially growing dilation.

    Args:
        num_inputs: Number of input channels (feature dimension).
        num_channels: List whose length determines the network depth and
            whose values set the channel width at each level.
        kernel_size: Kernel size for every temporal block.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[TemporalBlock] = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a temporal sequence.

        Args:
            x: Input tensor of shape ``[batch, num_inputs, time]``.

        Returns:
            Output tensor of shape ``[batch, num_channels[-1], time]``.
        """
        return self.network(x)  # [batch, num_channels[-1], time]
