"""Multi-scale temporal encoder.

Fuses four parallel branches — WaveNet, TCN, Transformer, and LSTM — via
learned self-attention to produce a fixed-size temporal representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from btc_prediction.models.components.attention import PositionalEncoding
from btc_prediction.models.components.tcn import TemporalConvNet
from btc_prediction.models.components.wavenet import WaveNet


class MultiScaleTemporalEncoder(nn.Module):
    """Encode a temporal sequence through four parallel branches and fuse.

    Branches
    --------
    1. **WaveNet** – dilated causal convolutions with gated activations.
    2. **TCN** – weight-normalised dilated causal convolutions.
    3. **Transformer** – sinusoidal positional encoding + standard encoder.
    4. **LSTM** – bidirectional, two-layer recurrent network.

    The four branch outputs are concatenated and refined with multi-head
    self-attention followed by layer normalisation.

    Args:
        input_dim: Number of input features per time step.
        output_dim: Final embedding dimension (must be divisible by 4).
        num_heads: Number of attention heads for the fusion layer.
        num_transformer_layers: Depth of the Transformer encoder branch.
        dropout: Dropout probability used throughout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if output_dim % 4 != 0:
            raise ValueError(f"output_dim must be divisible by 4, got {output_dim}")

        self.output_dim = output_dim
        branch_dim = output_dim // 4  # 128 when output_dim=512

        # ----- branch 1: WaveNet -----
        skip_channels = 256
        self.wavenet = WaveNet(
            in_channels=input_dim,
            residual_channels=32,
            dilation_channels=32,
            skip_channels=skip_channels,
            num_layers=10,
            num_blocks=3,
        )
        self.wavenet_pool = nn.AdaptiveAvgPool1d(1)
        self.wavenet_proj = nn.Linear(skip_channels, branch_dim)

        # ----- branch 2: TCN -----
        tcn_channels = [64, 128, 256]
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=tcn_channels,
            dropout=dropout,
        )
        self.tcn_proj = nn.Linear(tcn_channels[-1], branch_dim)

        # ----- branch 3: Transformer -----
        d_model = branch_dim  # project input to branch_dim
        self.transformer_input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers,
        )
        self.transformer_proj = nn.Linear(d_model, branch_dim)

        # ----- branch 4: LSTM -----
        lstm_hidden = 128
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # Bidirectional → 2 * lstm_hidden for last hidden (concat fwd + bwd)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, branch_dim)

        # ----- fusion -----
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(output_dim)

        self._init_weights()

    # --------------------------------------------------------------------- #
    def _init_weights(self) -> None:
        """Initialise linear projection weights."""
        for module in (
            self.wavenet_proj,
            self.tcn_proj,
            self.transformer_input_proj,
            self.transformer_proj,
            self.lstm_proj,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a temporal sequence through four branches and fuse.

        Args:
            x: Input tensor of shape ``[batch, seq_len, input_dim]``.

        Returns:
            Fused embedding of shape ``[batch, output_dim]``.
        """
        # --- WaveNet branch ---
        # WaveNet expects [batch, channels, time]
        x_conv = x.transpose(1, 2)  # [B, input_dim, T]
        wn_out = self.wavenet(x_conv)  # [B, skip_channels, T]
        wn_out = self.wavenet_pool(wn_out).squeeze(-1)  # [B, skip_channels]
        wn_emb = self.wavenet_proj(wn_out)  # [B, branch_dim]

        # --- TCN branch ---
        tcn_out = self.tcn(x_conv)  # [B, tcn_channels[-1], T]
        tcn_last = tcn_out[:, :, -1]  # [B, tcn_channels[-1]]
        tcn_emb = self.tcn_proj(tcn_last)  # [B, branch_dim]

        # --- Transformer branch ---
        tf_in = self.transformer_input_proj(x)  # [B, T, d_model]
        tf_in = self.positional_encoding(tf_in)  # [B, T, d_model]
        tf_out = self.transformer_encoder(tf_in)  # [B, T, d_model]
        tf_pool = tf_out.mean(dim=1)  # [B, d_model]
        tf_emb = self.transformer_proj(tf_pool)  # [B, branch_dim]

        # --- LSTM branch ---
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers*2, B, hidden]
        # Concatenate last-layer forward and backward hidden states
        h_fwd = h_n[-2]  # [B, hidden]
        h_bwd = h_n[-1]  # [B, hidden]
        lstm_emb = self.lstm_proj(
            torch.cat([h_fwd, h_bwd], dim=-1),
        )  # [B, branch_dim]

        # --- fusion ---
        # Concatenate branch outputs → [B, output_dim]
        fused = torch.cat([wn_emb, tcn_emb, tf_emb, lstm_emb], dim=-1)
        # Self-attention over a single-token sequence
        fused = fused.unsqueeze(1)  # [B, 1, output_dim]
        attn_out, _ = self.fusion_attention(
            fused, fused, fused,
        )  # [B, 1, output_dim]
        out = self.layer_norm(fused + attn_out)  # [B, 1, output_dim]
        return out.squeeze(1)  # [B, output_dim]
