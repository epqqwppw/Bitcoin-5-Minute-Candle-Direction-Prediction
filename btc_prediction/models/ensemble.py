"""Ensemble prediction aggregator.

Combines predictions from multiple model instances using learned,
regime-aware weights so that the ensemble adapts to current market
conditions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnsemblePredictor(nn.Module):
    """Dynamically-weighted ensemble of prediction models.

    Each member model contributes a prediction dictionary.  A small network
    maps per-model performance metrics and the current regime distribution to
    per-model weights which are used to produce an aggregated prediction.

    Args:
        num_models: Number of models in the ensemble.
        num_regimes: Number of market regimes.
        metrics_per_model: Number of tracked performance metrics per model.
    """

    def __init__(
        self,
        num_models: int = 10,
        num_regimes: int = 5,
        metrics_per_model: int = 3,
    ) -> None:
        super().__init__()
        self.num_models = num_models
        self.num_regimes = num_regimes
        self.metrics_per_model = metrics_per_model

        # Network that turns per-model metrics into base weights
        self.weight_network = nn.Sequential(
            nn.Linear(num_models * metrics_per_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_models),
            nn.Softmax(dim=-1),
        )

        # Learnable regime-specific adjustments applied on top of base weights
        self.regime_adjustments = nn.Parameter(
            torch.zeros(num_regimes, num_models),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.weight_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #
    #  Weight computation                                                 #
    # ------------------------------------------------------------------ #

    def compute_weights(
        self,
        performance_metrics: torch.Tensor,
        regime_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-model weights conditioned on performance and regime.

        Args:
            performance_metrics: Flattened metrics ``[num_models * metrics_per_model]``
                or ``[batch, num_models * metrics_per_model]``.
            regime_probs: Regime distribution ``[num_regimes]`` or ``[batch, num_regimes]``.

        Returns:
            Normalised weights ``[num_models]`` or ``[batch, num_models]``.
        """
        squeeze = False
        if performance_metrics.dim() == 1:
            performance_metrics = performance_metrics.unsqueeze(0)
            regime_probs = regime_probs.unsqueeze(0)
            squeeze = True

        base_weights = self.weight_network(performance_metrics)  # [batch, num_models]

        # regime_probs @ regime_adjustments → [batch, num_models]
        regime_adj = torch.matmul(regime_probs, self.regime_adjustments)
        adjusted = base_weights + regime_adj
        weights = torch.softmax(adjusted, dim=-1)  # re-normalise

        if squeeze:
            weights = weights.squeeze(0)
        return weights

    # ------------------------------------------------------------------ #
    #  Prediction aggregation                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def aggregate_predictions(
        predictions: list[dict[str, torch.Tensor]],
        weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Aggregate predictions from multiple models.

        Args:
            predictions: List of prediction dictionaries (one per model).
            weights: Per-model weights ``[num_models]`` or ``[batch, num_models]``.

        Returns:
            Aggregated prediction dictionary with the same keys as each input.
        """
        keys_to_avg = [
            "direction_probs",
            "quantiles",
            "volatility",
            "confidence",
            "uncertainty",
        ]
        result: dict[str, torch.Tensor] = {}

        # Ensure weights has shape suitable for broadcasting
        if weights.dim() == 1:
            w = weights  # [num_models]
        else:
            w = weights  # [batch, num_models]

        for key in keys_to_avg:
            stacked = torch.stack(
                [p[key] for p in predictions], dim=0,
            )  # [num_models, batch, *]
            if w.dim() == 1:
                # w[:, None, ...] → broadcast over batch and feature dims
                shape = [len(predictions)] + [1] * (stacked.dim() - 1)
                weighted = stacked * w.view(shape)
            else:
                # w is [batch, num_models] → need [num_models, batch, 1, ...]
                extra = stacked.dim() - 2  # dims after batch
                w_view = w.t().unsqueeze(-1) if extra == 1 else w.t()
                for _ in range(extra - 1):
                    w_view = w_view.unsqueeze(-1)
                if extra == 0:
                    w_view = w.t()  # [num_models, batch]
                weighted = stacked * w_view
            result[key] = weighted.sum(dim=0)

        # direction_logits: weighted sum
        if "direction_logits" in predictions[0]:
            logits_stack = torch.stack(
                [p["direction_logits"] for p in predictions], dim=0,
            )
            if w.dim() == 1:
                shape = [len(predictions)] + [1] * (logits_stack.dim() - 1)
                result["direction_logits"] = (logits_stack * w.view(shape)).sum(dim=0)
            else:
                w_view = w.t().unsqueeze(-1)
                result["direction_logits"] = (logits_stack * w_view).sum(dim=0)

        return result

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        predictions: list[dict[str, torch.Tensor]],
        performance_metrics: torch.Tensor,
        regime_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute ensemble weights and aggregate predictions.

        Args:
            predictions: Per-model prediction dictionaries.
            performance_metrics: Flattened model metrics
                ``[batch, num_models * metrics_per_model]``.
            regime_probs: Regime distribution ``[batch, num_regimes]``.

        Returns:
            Aggregated prediction dictionary.
        """
        weights = self.compute_weights(performance_metrics, regime_probs)
        return self.aggregate_predictions(predictions, weights)
