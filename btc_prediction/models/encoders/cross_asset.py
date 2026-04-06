"""Cross-asset encoder.

Estimates dynamic correlations between assets, encodes each asset's
temporal evolution independently, and fuses cross-asset information with
cross-attention and optional regime-conditional gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicCorrelationNet(nn.Module):
    """Estimate a time-varying correlation matrix from windowed returns.

    A small MLP maps a flattened window of multi-asset returns to the
    upper-triangular entries of a correlation matrix, which is then
    symmetrised and shifted to ensure unit diagonal.

    Args:
        num_assets: Number of assets.
        window_size: Look-back window in time steps.
        hidden_dim: Width of the hidden layer.
    """

    def __init__(
        self,
        num_assets: int,
        window_size: int = 10,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.window_size = window_size
        num_upper = num_assets * (num_assets - 1) // 2

        self.net = nn.Sequential(
            nn.Linear(num_assets * window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_upper),
            nn.Tanh(),  # correlations ∈ (-1, 1)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, windowed_returns: torch.Tensor) -> torch.Tensor:
        """Estimate a correlation matrix.

        Args:
            windowed_returns: ``[batch, window_size, num_assets]``

        Returns:
            Correlation matrix ``[batch, num_assets, num_assets]``.
        """
        batch = windowed_returns.size(0)
        flat = windowed_returns.reshape(batch, -1)  # [B, window * assets]
        upper = self.net(flat)  # [B, num_upper]

        # Build symmetric matrix from upper-triangular entries
        corr = torch.zeros(
            batch, self.num_assets, self.num_assets,
            device=upper.device, dtype=upper.dtype,
        )
        idx = torch.triu_indices(self.num_assets, self.num_assets, offset=1)
        corr[:, idx[0], idx[1]] = upper
        corr = corr + corr.transpose(1, 2)  # symmetrise

        # Unit diagonal
        eye = torch.eye(
            self.num_assets, device=upper.device, dtype=upper.dtype,
        ).unsqueeze(0)
        corr = corr + eye
        return corr  # [B, num_assets, num_assets]


class CrossAssetEncoder(nn.Module):
    """Encode multi-asset temporal data with dynamic correlations.

    Architecture
    ------------
    1. **DynamicCorrelationNet** – estimates pairwise asset correlations.
    2. **Per-asset LSTMs** – encode each asset's time series independently.
    3. **Cross-attention** – assets attend to one another, modulated by
       the estimated correlation matrix.
    4. **Regime gates** – optionally scale embeddings by regime probabilities.

    Args:
        num_assets: Number of assets (e.g. BTC, ETH, SPX, DXY, Gold).
        asset_dim: Feature dimension per asset per time step.
        output_dim: Final embedding dimension.
        num_regimes: Number of market-regime categories for gating.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_assets: int = 5,
        asset_dim: int = 16,
        output_dim: int = 256,
        num_regimes: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.asset_dim = asset_dim
        self.output_dim = output_dim
        self.num_regimes = num_regimes

        lstm_hidden = 64

        # Dynamic correlation estimation
        self.correlation_net = DynamicCorrelationNet(
            num_assets=num_assets, window_size=10, hidden_dim=64,
        )

        # Per-asset LSTM encoders
        self.asset_encoders = nn.ModuleList([
            nn.LSTM(
                input_size=asset_dim,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
            )
            for _ in range(num_assets)
        ])

        # Cross-attention between asset embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(lstm_hidden)

        # Regime-conditional gates
        self.regime_gates = nn.ModuleList([
            nn.Linear(num_regimes, lstm_hidden) for _ in range(num_assets)
        ])
        self.default_regime_proj = nn.Linear(lstm_hidden, lstm_hidden)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden * num_assets + lstm_hidden, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    # --------------------------------------------------------------------- #
    def _init_weights(self) -> None:
        """Initialise linear/LSTM weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        asset_data: torch.Tensor,
        regime_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode cross-asset data.

        Args:
            asset_data: ``[batch, seq_len, num_assets * asset_dim]``.
                Features for all assets concatenated along the last axis.
            regime_probs: Optional regime probability vector
                ``[batch, num_regimes]``.  When *None*, a learned default
                gating is applied.

        Returns:
            Cross-asset embedding of shape ``[batch, output_dim]``.
        """
        batch, seq_len, _ = asset_data.shape

        # Split into per-asset tensors → list of [B, T, asset_dim]
        asset_tensors = asset_data.view(
            batch, seq_len, self.num_assets, self.asset_dim,
        ).unbind(dim=2)

        # --- Per-asset encoding ---
        asset_embeds: list[torch.Tensor] = []
        for tensor, encoder in zip(
            asset_tensors, self.asset_encoders, strict=True,
        ):
            _, (h_n, _) = encoder(tensor)  # h_n: [1, B, hidden]
            asset_embeds.append(h_n.squeeze(0))  # [B, hidden]

        # Stack → [B, num_assets, hidden]
        asset_stack = torch.stack(asset_embeds, dim=1)

        # --- Dynamic correlation ---
        # Use last `window_size` steps of first feature per asset as returns
        window = min(seq_len, self.correlation_net.window_size)
        # Extract one "return-like" feature per asset (first feature)
        returns = asset_data[:, -window:, :].view(
            batch, window, self.num_assets, self.asset_dim,
        )[:, :, :, 0]  # [B, window, num_assets]
        if window < self.correlation_net.window_size:
            pad_size = self.correlation_net.window_size - window
            pad = torch.zeros(
                batch, pad_size, self.num_assets,
                device=asset_data.device, dtype=asset_data.dtype,
            )
            returns = torch.cat([pad, returns], dim=1)
        corr_matrix = self.correlation_net(returns)  # [B, A, A]

        # --- Cross-attention modulated by correlation ---
        # Use correlation as attention bias
        corr_bias = corr_matrix.unsqueeze(1).expand(
            -1, self.cross_attention.num_heads, -1, -1,
        ).reshape(batch * self.cross_attention.num_heads,
                  self.num_assets, self.num_assets)
        attn_out, _ = self.cross_attention(
            asset_stack, asset_stack, asset_stack,
            attn_mask=corr_bias,
        )  # [B, num_assets, hidden]
        asset_attended = self.cross_norm(asset_stack + attn_out)

        # --- Regime gating ---
        gated_embeds: list[torch.Tensor] = []
        for i in range(self.num_assets):
            emb = asset_attended[:, i, :]  # [B, hidden]
            if regime_probs is not None:
                gate = torch.sigmoid(
                    self.regime_gates[i](regime_probs),
                )  # [B, hidden]
            else:
                gate = torch.sigmoid(
                    self.default_regime_proj(emb),
                )  # [B, hidden]
            gated_embeds.append(emb * gate)  # [B, hidden]

        # --- Aggregate correlation info ---
        corr_summary = corr_matrix.mean(dim=-1)  # [B, num_assets]
        corr_weighted = torch.einsum(
            "ba,bah->bh", corr_summary, asset_attended,
        )  # [B, hidden]

        # --- Final projection ---
        combined = torch.cat(
            [*gated_embeds, corr_weighted], dim=-1,
        )  # [B, num_assets * hidden + hidden]
        return self.output_proj(combined)  # [B, output_dim]
