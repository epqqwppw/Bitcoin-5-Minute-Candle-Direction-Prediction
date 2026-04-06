"""Hierarchical multi-modal fusion module.

Merges encoder embeddings through three successive levels — self-attention,
cross-modal attention, and regime-adaptive gating — into a single fused
representation suitable for the prediction heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from btc_prediction.models.components.attention import CrossModalAttention

# Common projection dimension used across all fusion levels.
_COMMON_DIM: int = 256


class HierarchicalFusion(nn.Module):
    """Three-level hierarchical fusion of multi-modal embeddings.

    **Level 1** – Per-embedding self-attention (after projection to a common
    dimensionality).

    **Level 2** – Pairwise cross-modal attention across all embedding pairs,
    enabling every modality to absorb information from every other.

    **Level 3** – Regime-adaptive gating that re-weights each embedding
    according to the current market regime.

    The gated embeddings are concatenated, layer-normalised and projected to
    ``output_dim``.

    Args:
        embedding_dims: Dimensionality of each incoming encoder embedding
            (e.g. ``[512, 256, 256, 256, 256]``).
        output_dim: Dimensionality of the fused output vector.
        num_heads: Number of attention heads used in Levels 1 and 2.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embedding_dims: list[int],
        output_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_embeddings = len(embedding_dims)
        self.output_dim = output_dim
        self.common_dim = _COMMON_DIM

        # --- Projection layers: map each embedding to common_dim ---
        self.projections = nn.ModuleList(
            [nn.Linear(dim, self.common_dim) for dim in embedding_dims]
        )

        # --- Level 1: self-attention within each (projected) embedding ---
        self.self_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.common_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(self.num_embeddings)
            ]
        )
        self.self_attn_norms = nn.ModuleList(
            [nn.LayerNorm(self.common_dim) for _ in range(self.num_embeddings)]
        )

        # --- Level 2: cross-modal attention between all pairs ---
        self.cross_attentions = nn.ModuleDict(
            {
                f"{i}_{j}": CrossModalAttention(
                    d_model=self.common_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for i in range(self.num_embeddings)
                for j in range(self.num_embeddings)
                if i != j
            }
        )

        # --- Level 3: regime-adaptive gating ---
        self.gating_network: nn.Module = nn.Sequential(
            nn.Linear(5, self.num_embeddings),  # num_regimes defaults to 5
            nn.Sigmoid(),
        )

        # --- Final projection ---
        concat_dim = self.common_dim * self.num_embeddings
        self.output_norm = nn.LayerNorm(concat_dim)
        self.output_proj = nn.Linear(concat_dim, output_dim)

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        embeddings: list[torch.Tensor],
        regime_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multi-modal embeddings hierarchically.

        Args:
            embeddings: List of encoder outputs, each ``[batch, emb_dim_i]``.
            regime_probs: Regime probability distribution ``[batch, num_regimes]``.

        Returns:
            Fused representation of shape ``[batch, output_dim]``.
        """
        batch = embeddings[0].size(0)

        # Project to common dim and reshape for attention: [batch, 1, common_dim]
        projected: list[torch.Tensor] = [
            proj(emb).unsqueeze(1) for proj, emb in zip(self.projections, embeddings, strict=True)
        ]

        # --- Level 1: per-embedding self-attention ---
        attended: list[torch.Tensor] = []
        for idx, (x, attn, norm) in enumerate(
            zip(projected, self.self_attentions, self.self_attn_norms, strict=True)
        ):
            out, _ = attn(x, x, x)  # [batch, 1, common_dim]
            attended.append(norm(x + out))

        # --- Level 2: pairwise cross-modal attention ---
        cross_outputs: list[list[torch.Tensor]] = [[] for _ in range(self.num_embeddings)]
        for i in range(self.num_embeddings):
            for j in range(self.num_embeddings):
                if i == j:
                    continue
                key = f"{i}_{j}"
                cross_out = self.cross_attentions[key](attended[i], attended[j])
                cross_outputs[i].append(cross_out)  # [batch, 1, common_dim]

        # Average cross-attended contributions for each embedding
        fused_embeds: list[torch.Tensor] = []
        for i in range(self.num_embeddings):
            if cross_outputs[i]:
                avg_cross = torch.stack(cross_outputs[i], dim=0).mean(dim=0)
            else:
                avg_cross = attended[i]
            fused_embeds.append(avg_cross.squeeze(1))  # [batch, common_dim]

        # --- Level 3: regime-adaptive gating ---
        gate_weights = self.gating_network(regime_probs)  # [batch, num_embeddings]

        gated: list[torch.Tensor] = []
        for i in range(self.num_embeddings):
            # gate_weights[:, i] → [batch, 1] broadcast multiply
            gated.append(fused_embeds[i] * gate_weights[:, i].unsqueeze(-1))

        # --- Concatenate + project ---
        concat = torch.cat(gated, dim=-1)  # [batch, common_dim * num_embeddings]
        normed = self.output_norm(concat)
        return self.output_proj(normed)  # [batch, output_dim]
