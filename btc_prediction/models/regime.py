"""Market regime detection module.

Classifies the current market state into one of several regimes
(bull, bear, high-volatility, low-volatility, consolidation) and produces
a regime-conditioned embedding for downstream fusion and prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegimeDetector(nn.Module):
    """Detect the prevailing market regime from a fused market-state vector.

    The detector extracts features, classifies into one of ``num_regimes``
    regimes, and returns a regime-probability distribution together with a
    regime-conditioned embedding (weighted sum of learnable per-regime
    parameter vectors).

    Args:
        input_dim: Dimensionality of the incoming market-state vector.
        num_regimes: Number of distinct market regimes.
        dropout: Dropout probability used between feature layers.
    """

    REGIMES: list[str] = [
        "bull",
        "bear",
        "high_vol",
        "low_vol",
        "consolidation",
    ]

    def __init__(
        self,
        input_dim: int,
        num_regimes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.regimes = list(self.REGIMES[:num_regimes])

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, num_regimes),
            nn.Softmax(dim=-1),
        )

        # Learnable regime-specific parameter vectors
        self.regime_params = nn.ParameterDict(
            {
                regime: nn.Parameter(torch.randn(128))
                for regime in self.regimes
            }
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.feature_extractor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, market_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Detect the market regime and produce a regime embedding.

        Args:
            market_state: Market-state vector of shape ``[batch, input_dim]``.

        Returns:
            Dictionary with:
            - ``regime_probs``: ``[batch, num_regimes]`` probability distribution.
            - ``regime_embedding``: ``[batch, 128]`` weighted regime embedding.
        """
        features = self.feature_extractor(market_state)  # [batch, 128]
        regime_probs = self.classifier(features)  # [batch, num_regimes]

        # Stack regime parameter vectors → [num_regimes, 128]
        param_stack = torch.stack(
            [self.regime_params[r] for r in self.regimes], dim=0,
        )

        # Weighted sum: [batch, num_regimes] @ [num_regimes, 128] → [batch, 128]
        regime_embedding = torch.matmul(regime_probs, param_stack)

        return {
            "regime_probs": regime_probs,
            "regime_embedding": regime_embedding,
        }
