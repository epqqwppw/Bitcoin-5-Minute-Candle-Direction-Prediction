"""End-to-end BTC candle direction predictor.

Composes five specialised encoders, a regime detector, hierarchical fusion,
and multi-task prediction heads into a single differentiable pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from btc_prediction.config.settings import ModelConfig
from btc_prediction.models.encoders import (
    CrossAssetEncoder,
    MultiScaleTemporalEncoder,
    OnChainEncoder,
    OrderBookEncoder,
    SentimentEncoder,
)
from btc_prediction.models.fusion import HierarchicalFusion
from btc_prediction.models.heads import PredictionHeads
from btc_prediction.models.regime import RegimeDetector


class BTCPredictor(nn.Module):
    """Full prediction pipeline for Bitcoin 5-minute candle direction.

    Orchestrates encoding, regime detection, hierarchical fusion, and
    multi-task prediction in a single ``forward`` call.

    Args:
        config: Model architecture configuration (dimensions, heads, etc.).
        temporal_input_dim: Number of features in the temporal (price) stream.
        orderbook_levels: Number of order-book depth levels.
        onchain_input_dim: Number of raw on-chain features.
        num_assets: Number of cross-asset instruments.
        asset_dim: Feature dimensionality per cross-asset instrument.
        sentiment_input_dim: Dimensionality of pre-computed sentiment features.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        temporal_input_dim: int = 64,
        orderbook_levels: int = 10,
        onchain_input_dim: int = 32,
        num_assets: int = 5,
        asset_dim: int = 16,
        sentiment_input_dim: int = 32,
    ) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # ---- Encoders ----
        self.temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=temporal_input_dim,
            output_dim=config.temporal_dim,
            num_heads=config.num_heads,
            num_transformer_layers=config.num_transformer_layers,
            dropout=config.dropout,
        )
        self.orderbook_encoder = OrderBookEncoder(
            num_levels=orderbook_levels,
            output_dim=config.orderbook_dim,
            dropout=config.dropout,
        )
        self.onchain_encoder = OnChainEncoder(
            input_dim=onchain_input_dim,
            output_dim=config.onchain_dim,
            dropout=config.dropout,
        )
        self.cross_asset_encoder = CrossAssetEncoder(
            num_assets=num_assets,
            asset_dim=asset_dim,
            output_dim=config.cross_asset_dim,
            num_regimes=config.num_regimes,
            dropout=config.dropout,
        )
        self.sentiment_encoder = SentimentEncoder(
            input_dim=sentiment_input_dim,
            output_dim=config.sentiment_dim,
            dropout=config.dropout,
        )

        # ---- Regime detector ----
        # Input dim = sum of all encoder output dims
        encoder_dims = [
            config.temporal_dim,
            config.orderbook_dim,
            config.onchain_dim,
            config.cross_asset_dim,
            config.sentiment_dim,
        ]
        regime_input_dim = sum(encoder_dims)
        self.regime_detector = RegimeDetector(
            input_dim=regime_input_dim,
            num_regimes=config.num_regimes,
            dropout=config.dropout,
        )

        # ---- Hierarchical fusion ----
        self.fusion = HierarchicalFusion(
            embedding_dims=encoder_dims,
            output_dim=config.fused_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # ---- Prediction heads ----
        self.prediction_heads = PredictionHeads(
            input_dim=config.fused_dim,
            num_classes=3,
            num_quantiles=config.num_quantiles,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        price_data: torch.Tensor,
        orderbook_data: torch.Tensor,
        onchain_data: torch.Tensor,
        cross_asset_data: torch.Tensor,
        sentiment_data: torch.Tensor,
        volatility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full prediction pipeline.

        Args:
            price_data: ``[batch, seq, temporal_features]``.
            orderbook_data: ``[batch, seq, orderbook_features]``.
            onchain_data: ``[batch, seq, onchain_features]``.
            cross_asset_data: ``[batch, seq, cross_asset_features]``.
            sentiment_data: ``[batch, seq, sentiment_features]``.
            volatility: Optional ``[batch, 1]`` or ``[batch]`` volatility
                context fed to the sentiment encoder.

        Returns:
            Dictionary containing all prediction outputs plus regime info:
            ``direction_logits``, ``direction_probs``, ``quantiles``,
            ``volatility``, ``confidence``, ``uncertainty``, ``auxiliary``,
            ``regime_probs``, ``regime_embedding``.
        """
        # 1. Encode each data stream → [batch, dim]
        temporal_emb = self.temporal_encoder(price_data)
        orderbook_emb = self.orderbook_encoder(orderbook_data)
        onchain_emb = self.onchain_encoder(onchain_data)
        sentiment_emb = self.sentiment_encoder(sentiment_data, volatility=volatility)

        # Initial regime estimate so the cross-asset encoder can condition
        # on regime_probs.  Cross-asset embedding is not yet available, so we
        # substitute a zero placeholder of matching shape.
        cross_asset_placeholder = torch.zeros(
            temporal_emb.size(0), self.config.cross_asset_dim,
            device=temporal_emb.device, dtype=temporal_emb.dtype,
        )
        initial_state = torch.cat(
            [temporal_emb, orderbook_emb, onchain_emb,
             cross_asset_placeholder, sentiment_emb],
            dim=-1,
        )
        initial_regime = self.regime_detector(initial_state)

        cross_asset_emb = self.cross_asset_encoder(
            cross_asset_data,
            regime_probs=initial_regime["regime_probs"],
        )

        # 2. Full regime detection with all embeddings
        market_state = torch.cat(
            [temporal_emb, orderbook_emb, onchain_emb,
             cross_asset_emb, sentiment_emb],
            dim=-1,
        )
        regime_out = self.regime_detector(market_state)

        # 3. Hierarchical fusion
        embeddings = [
            temporal_emb,
            orderbook_emb,
            onchain_emb,
            cross_asset_emb,
            sentiment_emb,
        ]
        fused = self.fusion(embeddings, regime_out["regime_probs"])

        # 4. Multi-task prediction
        predictions = self.prediction_heads(fused)

        # 5. Attach regime info
        predictions["regime_probs"] = regime_out["regime_probs"]
        predictions["regime_embedding"] = regime_out["regime_embedding"]

        return predictions
