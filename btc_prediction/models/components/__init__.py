"""Reusable neural network components for temporal and graph-based models.

Exports
-------
- WaveNet components: ``CausalConv1d``, ``WaveNetResidualBlock``, ``WaveNet``
- TCN components: ``TemporalBlock``, ``TemporalConvNet``
- Attention components: ``PositionalEncoding``, ``MultiScaleAttention``,
  ``CrossModalAttention``, ``GatedFusion``
- GNN components: ``GraphConvLayer``, ``TemporalGraphConv``, ``OrderBookGraphBuilder``
"""

from __future__ import annotations

from btc_prediction.models.components.attention import (
    CrossModalAttention,
    GatedFusion,
    MultiScaleAttention,
    PositionalEncoding,
)
from btc_prediction.models.components.gnn import (
    GraphConvLayer,
    OrderBookGraphBuilder,
    TemporalGraphConv,
)
from btc_prediction.models.components.tcn import TemporalBlock, TemporalConvNet
from btc_prediction.models.components.wavenet import (
    CausalConv1d,
    WaveNet,
    WaveNetResidualBlock,
)

__all__ = [
    "CausalConv1d",
    "CrossModalAttention",
    "GatedFusion",
    "GraphConvLayer",
    "MultiScaleAttention",
    "OrderBookGraphBuilder",
    "PositionalEncoding",
    "TemporalBlock",
    "TemporalConvNet",
    "TemporalGraphConv",
    "WaveNet",
    "WaveNetResidualBlock",
]
