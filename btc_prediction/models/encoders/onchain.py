"""On-chain data encoder.

Combines an LSTM for trend extraction, metric-specific MLPs, attention-based
metric importance weighting, and bilinear feature interaction to produce a
rich representation of on-chain blockchain metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OnChainEncoder(nn.Module):
    """Encode on-chain metrics into a fixed-size embedding.

    Architecture
    ------------
    1. **LSTM** – extracts temporal trends from the full input.
    2. **Metric-specific MLPs** – each metric slice is processed independently
       by a dedicated two-layer MLP.
    3. **Attention** – multi-head attention computes importance weights over
       the metric embeddings.
    4. **Bilinear interaction** – captures pairwise feature cross-terms.
    5. Final projection to ``output_dim``.

    Args:
        input_dim: Total number of input features per time step.
        output_dim: Final embedding dimension.
        num_metrics: Number of on-chain metric groups. ``input_dim`` is split
            evenly across these groups; any remainder is absorbed by the last
            group.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        num_metrics: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_metrics = num_metrics

        # Compute per-metric feature slices
        base_width = input_dim // num_metrics
        remainder = input_dim % num_metrics
        self.metric_widths: list[int] = [
            base_width + (1 if i < remainder else 0) for i in range(num_metrics)
        ]

        lstm_hidden = 128

        # ----- LSTM trend extractor -----
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # ----- metric-specific MLPs -----
        self.metric_mlps = nn.ModuleList()
        for width in self.metric_widths:
            self.metric_mlps.append(
                nn.Sequential(
                    nn.Linear(width, lstm_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(lstm_hidden, 64),
                ),
            )

        # ----- attention for metric importance -----
        metric_emb_dim = lstm_hidden  # query from LSTM, keys from metric MLPs
        self.metric_key_proj = nn.Linear(64, metric_emb_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=metric_emb_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(metric_emb_dim)

        # ----- bilinear interaction -----
        self.bilinear = nn.Bilinear(metric_emb_dim, metric_emb_dim, metric_emb_dim)

        # ----- output projection -----
        # LSTM hidden + attended metrics + bilinear cross-term
        combined_dim = lstm_hidden + metric_emb_dim + metric_emb_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    # --------------------------------------------------------------------- #
    def _init_weights(self) -> None:
        """Initialise linear/bilinear weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Bilinear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode on-chain metrics.

        Args:
            x: Input tensor of shape ``[batch, seq_len, input_dim]``.

        Returns:
            On-chain embedding of shape ``[batch, output_dim]``.
        """
        batch = x.size(0)

        # --- LSTM trend ---
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: [B, T, hidden]
        lstm_last = h_n[-1]  # [B, hidden]  (last layer, last step)

        # --- Metric-specific MLPs (applied to the last time step) ---
        x_last = x[:, -1, :]  # [B, input_dim]
        metric_embeds: list[torch.Tensor] = []
        offset = 0
        for width, mlp in zip(self.metric_widths, self.metric_mlps, strict=True):
            metric_slice = x_last[:, offset : offset + width]  # [B, width]
            metric_embeds.append(mlp(metric_slice))  # [B, 64]
            offset += width

        # Stack metric embeddings → [B, num_metrics, 64]
        metric_stack = torch.stack(metric_embeds, dim=1)

        # Project metric embeddings to LSTM hidden size for attention
        metric_keys = self.metric_key_proj(metric_stack)  # [B, num_metrics, hidden]

        # --- Attention: LSTM hidden queries metric embeddings ---
        query = lstm_last.unsqueeze(1)  # [B, 1, hidden]
        attn_out, _ = self.attention(
            query, metric_keys, metric_keys,
        )  # [B, 1, hidden]
        attn_out = self.attn_norm(query + attn_out).squeeze(1)  # [B, hidden]

        # --- Bilinear interaction ---
        cross = self.bilinear(lstm_last, attn_out)  # [B, hidden]

        # --- Combine and project ---
        combined = torch.cat(
            [lstm_last, attn_out, cross], dim=-1,
        )  # [B, hidden + hidden + hidden]
        return self.output_proj(combined)  # [B, output_dim]
