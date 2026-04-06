"""Attention mechanisms and fusion modules for multi-modal learning.

Provides sinusoidal positional encoding, multi-scale self-attention,
cross-modal attention, and gated fusion — building blocks used by the
ensemble prediction model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Adds fixed sin/cos positional signals to the input embeddings so that
    the model can exploit ordering information.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length to pre-compute.
        dropout: Dropout probability applied after the addition.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)              # [max_len, d_model]
        position = torch.arange(max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model // 2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Positionally-encoded tensor of shape ``[batch, seq_len, d_model]``.
        """
        x = x + self.pe[:, : x.size(1)]  # [batch, seq_len, d_model]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-scale self-attention
# ---------------------------------------------------------------------------


class MultiScaleAttention(nn.Module):
    """Multi-head attention augmented with learnable scale parameters.

    Each *scale* applies a separate linear projection before the shared
    multi-head attention, allowing the model to attend at different
    representational granularities.

    Args:
        d_model: Model / embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
        num_scales: Number of learnable scale projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_scales: int = 4,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales

        # Per-scale input projections
        self.scale_projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_scales)]
        )
        # Learnable importance weight per scale
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for proj in self.scale_projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale self-attention.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Attended tensor of shape ``[batch, seq_len, d_model]``.
        """
        # Weighted combination of scale projections
        weights = torch.softmax(self.scale_weights, dim=0)  # [num_scales]
        scaled = torch.zeros_like(x)  # [batch, seq_len, d_model]
        for i, proj in enumerate(self.scale_projections):
            scaled = scaled + weights[i] * proj(x)

        # Self-attention with residual + layer norm
        attn_out, _ = self.attention(scaled, scaled, scaled)  # [B, T, D]
        return self.layer_norm(x + attn_out)  # [batch, seq_len, d_model]


# ---------------------------------------------------------------------------
# Cross-modal attention
# ---------------------------------------------------------------------------


class CrossModalAttention(nn.Module):
    """Cross-attention between two modalities.

    The *query* modality attends to the *key/value* modality, enabling
    information flow from one data source to another.

    Args:
        d_model: Model / embedding dimension (must match both inputs).
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_kv = nn.LayerNorm(d_model)
        self.layer_norm_out = nn.LayerNorm(d_model)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend *query* to *key_value*.

        Args:
            query: Query tensor of shape ``[batch, seq_q, d_model]``.
            key_value: Key/value tensor of shape ``[batch, seq_kv, d_model]``.

        Returns:
            Cross-attended tensor of shape ``[batch, seq_q, d_model]``.
        """
        q = self.layer_norm_q(query)        # [batch, seq_q, d_model]
        kv = self.layer_norm_kv(key_value)  # [batch, seq_kv, d_model]

        attn_out, _ = self.attention(q, kv, kv)  # [batch, seq_q, d_model]
        return self.layer_norm_out(query + attn_out)  # [batch, seq_q, d_model]


# ---------------------------------------------------------------------------
# Gated fusion
# ---------------------------------------------------------------------------


class GatedFusion(nn.Module):
    """Gated fusion of multiple input tensors.

    Learns a softmax gate over *N* inputs and returns their weighted sum.

    Args:
        input_dim: Dimensionality of each input tensor (last dim).
        num_inputs: Number of tensors to fuse.
    """

    def __init__(self, input_dim: int, num_inputs: int) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        # One gating projection per input
        self.gate_projections = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_inputs)]
        )
        self.gate_norm = nn.LayerNorm(input_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for proj in self.gate_projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple tensors with learned gating.

        Args:
            inputs: List of *num_inputs* tensors, each of shape
                ``[batch, *, input_dim]``.

        Returns:
            Fused tensor of the same shape as each input.

        Raises:
            ValueError: If the number of inputs doesn't match ``num_inputs``.
        """
        if len(inputs) != self.num_inputs:
            msg = (
                f"Expected {self.num_inputs} inputs, got {len(inputs)}"
            )
            raise ValueError(msg)

        # Compute per-input gate logits  → [num_inputs, batch, *, input_dim]
        gate_logits = torch.stack(
            [proj(inp) for proj, inp in zip(self.gate_projections, inputs, strict=True)],
            dim=0,
        )
        gate_weights = torch.softmax(gate_logits, dim=0)  # normalise across inputs

        stacked = torch.stack(inputs, dim=0)  # [num_inputs, batch, *, input_dim]
        fused = (gate_weights * stacked).sum(dim=0)  # [batch, *, input_dim]
        return self.gate_norm(fused)
