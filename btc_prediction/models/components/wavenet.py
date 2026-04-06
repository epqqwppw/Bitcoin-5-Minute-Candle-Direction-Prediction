"""WaveNet architecture for temporal feature extraction.

Implements dilated causal convolutions with gated activations and skip
connections, following the architecture described in van den Oord et al.
(2016) — adapted for financial time-series modelling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1-D causal convolution that preserves the temporal dimension.

    Left-pads the input so that the output at time *t* depends only on
    inputs at times ≤ *t*.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Input tensor of shape ``[batch, channels, time]``.

        Returns:
            Output tensor of shape ``[batch, out_channels, time]``.
        """
        # Pad on the left (causal) side only
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)  # [batch, out_channels, time]


class WaveNetResidualBlock(nn.Module):
    """Single WaveNet residual block with gated activation.

    Architecture::

        dilated causal conv ──┬── tanh ──╮
                              └── σ   ───╯── × ── 1×1 conv ──┬── + input → residual
                                                              └────────── → skip

    Args:
        residual_channels: Width of the residual path.
        dilation_channels: Width of the dilated convolution output.
        skip_channels: Width of the skip-connection output.
        kernel_size: Kernel size for the dilated causal convolution.
        dilation: Dilation factor for this block.
    """

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        # Gated activation: filter (tanh) and gate (sigmoid) branches
        self.filter_conv = CausalConv1d(
            residual_channels, dilation_channels, kernel_size, dilation,
        )
        self.gate_conv = CausalConv1d(
            residual_channels, dilation_channels, kernel_size, dilation,
        )

        # 1×1 convolutions for residual and skip projections
        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1)

        nn.init.kaiming_normal_(self.residual_conv.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.skip_conv.weight, nonlinearity="relu")

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute residual and skip outputs.

        Args:
            x: Input tensor of shape ``[batch, residual_channels, time]``.

        Returns:
            Tuple of (residual, skip) tensors:
            - residual: ``[batch, residual_channels, time]``
            - skip: ``[batch, skip_channels, time]``
        """
        # Gated activation unit
        h_filter = torch.tanh(self.filter_conv(x))   # [batch, dilation_ch, time]
        h_gate = torch.sigmoid(self.gate_conv(x))     # [batch, dilation_ch, time]
        h = h_filter * h_gate                          # [batch, dilation_ch, time]

        # Residual and skip projections
        residual = self.residual_conv(h) + x           # [batch, residual_ch, time]
        skip = self.skip_conv(h)                       # [batch, skip_ch, time]
        return residual, skip


class WaveNet(nn.Module):
    """Full WaveNet encoder with stacked dilated causal convolution blocks.

    Produces a skip-connection aggregate suitable for downstream heads.
    Dilations grow as ``2^layer_idx`` within each block, and the pattern
    repeats for ``num_blocks`` blocks.

    Args:
        in_channels: Number of input feature channels.
        residual_channels: Width of the residual path.
        dilation_channels: Width of the dilated convolution.
        skip_channels: Width of the skip connections.
        num_layers: Number of layers per block (dilation 1 … 2^(num_layers-1)).
        num_blocks: Number of times to repeat the dilation pattern.
    """

    def __init__(
        self,
        in_channels: int,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        num_layers: int = 10,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        # Project raw features into the residual channel space
        self.input_conv = nn.Conv1d(in_channels, residual_channels, 1)
        nn.init.kaiming_normal_(self.input_conv.weight, nonlinearity="linear")

        self.residual_blocks = nn.ModuleList()
        for _block in range(num_blocks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.residual_blocks.append(
                    WaveNetResidualBlock(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        dilation=dilation,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a temporal sequence with stacked dilated causal convolutions.

        Args:
            x: Input tensor of shape ``[batch, in_channels, time]``.

        Returns:
            Aggregated skip-connection output of shape
            ``[batch, skip_channels, time]``.
        """
        h = self.input_conv(x)  # [batch, residual_channels, time]

        skip_sum = torch.zeros_like(h[:, :1, :]).expand(
            -1, self.residual_blocks[0].skip_conv.out_channels, -1,
        )  # [batch, skip_channels, time]

        for block in self.residual_blocks:
            h, skip = block(h)   # h: [B, residual_ch, T], skip: [B, skip_ch, T]
            skip_sum = skip_sum + skip

        return skip_sum  # [batch, skip_channels, time]
