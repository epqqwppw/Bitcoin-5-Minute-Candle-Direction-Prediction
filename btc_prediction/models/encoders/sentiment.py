"""Sentiment encoder.

Lightweight encoder for pre-computed sentiment features (not raw text).
Applies temporal attention pooling, a sentiment-specific MLP, and optional
volatility-adaptive weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SentimentEncoder(nn.Module):
    """Encode pre-computed sentiment features into a fixed-size vector.

    Architecture
    ------------
    1. **Temporal attention pooling** – learns which time windows carry the
       most predictive sentiment signal.
    2. **Sentiment MLP** – non-linear projection of the pooled features.
    3. **Volatility weighting** – learns to scale the embedding by a
       volatility-derived gate so that sentiment signal is amplified during
       high-volatility regimes.

    Args:
        input_dim: Dimension of pre-computed sentiment features per time step.
        output_dim: Final embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 32,
        output_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden = 128

        # ----- temporal attention pooling -----
        self.temporal_proj = nn.Linear(input_dim, hidden)
        self.temporal_attn = nn.Linear(hidden, 1)

        # ----- sentiment MLP -----
        self.sentiment_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- volatility weighting -----
        self.vol_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Sigmoid(),
        )
        self.vol_default = nn.Parameter(torch.zeros(1))  # learned default

        # ----- output projection -----
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    # --------------------------------------------------------------------- #
    def _init_weights(self) -> None:
        """Initialise linear weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        sentiment_features: torch.Tensor,
        volatility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode sentiment features.

        Args:
            sentiment_features: ``[batch, seq_len, input_dim]``.
                Pre-computed sentiment scores / embeddings per time window.
            volatility: Optional ``[batch, 1]`` or ``[batch]`` scalar
                representing current market volatility.  When *None* a
                learned default is used.

        Returns:
            Sentiment embedding of shape ``[batch, output_dim]``.
        """
        batch = sentiment_features.size(0)

        # --- Temporal attention pooling ---
        proj = torch.tanh(
            self.temporal_proj(sentiment_features),
        )  # [B, T, hidden]
        attn_logits = self.temporal_attn(proj).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, T]
        pooled = torch.einsum(
            "bt,btd->bd", attn_weights, sentiment_features,
        )  # [B, input_dim]

        # --- Sentiment MLP ---
        sent_emb = self.sentiment_mlp(pooled)  # [B, hidden]

        # --- Volatility weighting ---
        if volatility is not None:
            if volatility.dim() == 1:
                volatility = volatility.unsqueeze(-1)  # [B, 1]
            vol_gate = self.vol_proj(volatility)  # [B, hidden]
        else:
            vol_gate = self.vol_proj(
                self.vol_default.expand(batch, 1),
            )  # [B, hidden]

        weighted = sent_emb * vol_gate  # [B, hidden]

        # --- Output ---
        return self.output_proj(weighted)  # [B, output_dim]
