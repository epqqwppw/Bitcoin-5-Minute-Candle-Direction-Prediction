"""Encoder modules for multi-modal Bitcoin prediction.

Exports
-------
- ``MultiScaleTemporalEncoder`` – fuses WaveNet, TCN, Transformer, and LSTM
  branches for temporal feature extraction.
- ``OrderBookEncoder`` – graph-based encoder for limit-order-book snapshots.
- ``OnChainEncoder`` – LSTM + metric-specific MLPs for on-chain data.
- ``CrossAssetEncoder`` – dynamic correlation + per-asset LSTM + cross-attention.
- ``SentimentEncoder`` – lightweight encoder for pre-computed sentiment features.
"""

from __future__ import annotations

from btc_prediction.models.encoders.cross_asset import CrossAssetEncoder
from btc_prediction.models.encoders.onchain import OnChainEncoder
from btc_prediction.models.encoders.orderbook import OrderBookEncoder
from btc_prediction.models.encoders.sentiment import SentimentEncoder
from btc_prediction.models.encoders.temporal import MultiScaleTemporalEncoder

__all__ = [
    "CrossAssetEncoder",
    "MultiScaleTemporalEncoder",
    "OnChainEncoder",
    "OrderBookEncoder",
    "SentimentEncoder",
]
