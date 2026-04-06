"""Multi-task prediction heads.

Provides five parallel heads that consume a single fused representation and
emit direction logits, quantile forecasts, volatility estimates, confidence /
uncertainty scores, and auxiliary price-level predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHeads(nn.Module):
    """Multi-task prediction heads for BTC candle prediction.

    Five heads operate in parallel on the fused embedding:

    1. **Direction** – 3-class classification (down / neutral / up).
    2. **Quantile** – return-distribution quantile regression.
    3. **Volatility** – single positive-valued volatility estimate.
    4. **Confidence** – confidence and uncertainty scores in [0, 1].
    5. **Auxiliary** – high, low, and volume predictions.

    Args:
        input_dim: Dimensionality of the fused input vector.
        num_classes: Number of direction classes.
        num_quantiles: Number of quantile levels to predict.
        dropout: Dropout probability used inside each head.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        num_classes: int = 3,
        num_quantiles: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_quantiles = num_quantiles

        # Head 1 – Direction classification
        self.direction_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        # Head 2 – Return-distribution quantile regression
        self.quantile_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_quantiles),
        )

        # Head 3 – Volatility (positive via Softplus)
        self.volatility_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Softplus(),
        )

        # Head 4 – Confidence & uncertainty in [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

        # Head 5 – Auxiliary predictions (high, low, volume)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for head in (
            self.direction_head,
            self.quantile_head,
            self.volatility_head,
            self.confidence_head,
            self.auxiliary_head,
        ):
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        """Produce all prediction outputs from the fused representation.

        Args:
            fused: Fused embedding of shape ``[batch, input_dim]``.

        Returns:
            Dictionary containing:
            - ``direction_logits``: ``[batch, num_classes]`` raw logits.
            - ``direction_probs``: ``[batch, num_classes]`` softmax probabilities.
            - ``quantiles``: ``[batch, num_quantiles]``.
            - ``volatility``: ``[batch, 1]`` positive scalar.
            - ``confidence``: ``[batch, 1]``.
            - ``uncertainty``: ``[batch, 1]``.
            - ``auxiliary``: ``[batch, 3]`` (high, low, volume).
        """
        direction_logits = self.direction_head(fused)         # [batch, num_classes]
        direction_probs = torch.softmax(direction_logits, dim=-1)

        quantiles = self.quantile_head(fused)                 # [batch, num_quantiles]
        volatility = self.volatility_head(fused)              # [batch, 1]

        conf_unc = self.confidence_head(fused)                # [batch, 2]
        confidence = conf_unc[:, :1]                          # [batch, 1]
        uncertainty = conf_unc[:, 1:]                         # [batch, 1]

        auxiliary = self.auxiliary_head(fused)                 # [batch, 3]

        return {
            "direction_logits": direction_logits,
            "direction_probs": direction_probs,
            "quantiles": quantiles,
            "volatility": volatility,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "auxiliary": auxiliary,
        }
