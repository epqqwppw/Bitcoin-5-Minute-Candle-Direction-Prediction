"""Order-book encoder.

Combines graph neural networks with attention pooling and bid/ask imbalance
features to produce a fixed-size representation of limit-order-book data.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from btc_prediction.models.components.gnn import (
    OrderBookGraphBuilder,
    TemporalGraphConv,
)


class OrderBookEncoder(nn.Module):
    """Encode a sequence of order-book snapshots into a fixed-size vector.

    Pipeline
    --------
    1. Build a graph per snapshot via :class:`OrderBookGraphBuilder`.
    2. Process the temporal graph sequence with :class:`TemporalGraphConv`.
    3. Attention-pool over the node dimension.
    4. Compute bid/ask volume imbalance features through a small MLP.
    5. Concatenate graph embedding and imbalance features → linear → output.

    Args:
        num_levels: Number of price levels on each side (bid / ask).
        node_features: Feature dimension per node (e.g. price, volume,
            order_count, level_index).
        output_dim: Final embedding dimension.
        dropout: Dropout probability.
    """

    _GRAPH_BUILDER_NODE_DIM = 3  # OrderBookGraphBuilder produces 3-dim nodes

    def __init__(
        self,
        num_levels: int = 10,
        node_features: int = 4,
        output_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.node_features = node_features
        self.output_dim = output_dim
        self.num_nodes = 2 * num_levels  # bid + ask levels

        # Graph builder (stateless utility — no parameters)
        self.graph_builder = OrderBookGraphBuilder(num_levels=num_levels)

        # Optional projection when raw node_features ≠ builder's 3-dim output
        gcn_in = self._GRAPH_BUILDER_NODE_DIM
        if node_features != self._GRAPH_BUILDER_NODE_DIM:
            self.node_proj: nn.Module = nn.Linear(
                self._GRAPH_BUILDER_NODE_DIM, node_features,
            )
            gcn_in = node_features
        else:
            self.node_proj = nn.Identity()

        # Temporal GCN
        gcn_out = 64
        self.temporal_gcn = TemporalGraphConv(
            in_channels=gcn_in,
            out_channels=gcn_out,
            num_layers=3,
        )

        # Attention pool over nodes
        self.attn_query = nn.Linear(gcn_out, 1)
        nn.init.xavier_uniform_(self.attn_query.weight)

        # Imbalance subnet
        imbalance_feat_dim = 4  # raw imbalance, log ratio, abs diff, spread
        self.imbalance_mlp = nn.Sequential(
            nn.Linear(imbalance_feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(gcn_out + 32, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    # --------------------------------------------------------------------- #
    def _init_weights(self) -> None:
        """Initialise linear layers with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.attn_query:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _compute_imbalance(
        bid_volumes: torch.Tensor,
        ask_volumes: torch.Tensor,
    ) -> torch.Tensor:
        """Derive bid/ask imbalance features.

        Args:
            bid_volumes: ``[batch, num_levels]``
            ask_volumes: ``[batch, num_levels]``

        Returns:
            Imbalance feature vector ``[batch, 4]``.
        """
        bid_total = bid_volumes.sum(dim=-1, keepdim=True)  # [B, 1]
        ask_total = ask_volumes.sum(dim=-1, keepdim=True)  # [B, 1]
        total = (bid_total + ask_total).clamp(min=1e-8)

        imbalance = (bid_total - ask_total) / total
        log_ratio = torch.log(
            (bid_total + 1e-8) / (ask_total + 1e-8),
        )
        abs_diff = (bid_total - ask_total).abs() / total
        spread = ask_volumes[:, :1] - bid_volumes[:, :1]  # top-of-book

        return torch.cat(
            [imbalance, log_ratio, abs_diff, spread], dim=-1,
        )  # [B, 4]

    # --------------------------------------------------------------------- #
    def _reshape_flat_input(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        """Handle flattened ``[batch, seq_len, features]`` input.

        Reshapes to ``[batch, seq_len, num_nodes, node_features]`` by
        interpreting the last dimension as ``num_levels * 2 * node_features``.

        Args:
            x: Tensor of shape ``[batch, seq_len, features]``.

        Returns:
            Tensor of shape ``[batch, seq_len, num_nodes, node_features]``.
        """
        batch, seq_len, feat = x.shape
        expected = self.num_nodes * self.node_features
        if feat != expected:
            raise ValueError(
                f"Flattened feature dim {feat} does not match "
                f"num_nodes({self.num_nodes}) × node_features({self.node_features}) "
                f"= {expected}",
            )
        return x.view(batch, seq_len, self.num_nodes, self.node_features)

    # --------------------------------------------------------------------- #
    def forward(self, order_book_data: torch.Tensor) -> torch.Tensor:
        """Encode order-book snapshots.

        Args:
            order_book_data: Either
                - ``[batch, seq_len, num_levels * 2, node_features]``
                  (structured), or
                - ``[batch, seq_len, num_levels * 2 * node_features]``
                  (flattened).

        Returns:
            Order-book embedding of shape ``[batch, output_dim]``.
        """
        # Handle flattened input
        if order_book_data.dim() == 3:
            order_book_data = self._reshape_flat_input(order_book_data)
        # order_book_data: [B, T, N, F]

        batch, seq_len, num_nodes, _ = order_book_data.shape

        # --- Build graphs per snapshot ---
        # Split into bid / ask halves
        bid_data = order_book_data[:, :, : self.num_levels]  # [B, T, L, F]
        ask_data = order_book_data[:, :, self.num_levels :]  # [B, T, L, F]

        # Build graphs for every (batch, time) pair
        node_features_list: list[torch.Tensor] = []
        adj_list: list[torch.Tensor] = []
        for t in range(seq_len):
            bid_p = bid_data[:, t, :, 0]  # [B, L]
            bid_v = bid_data[:, t, :, 1]  # [B, L]
            ask_p = ask_data[:, t, :, 0]  # [B, L]
            ask_v = ask_data[:, t, :, 1]  # [B, L]
            nf, adj = self.graph_builder.build_graph(bid_p, bid_v, ask_p, ask_v)
            node_features_list.append(nf)  # [B, 2L, 3]
            adj_list.append(adj)  # [B, 2L, 2L]

        node_feats = torch.stack(node_features_list, dim=1)  # [B, T, 2L, 3]
        adj_mats = torch.stack(adj_list, dim=1)  # [B, T, 2L, 2L]

        # Optional node projection
        node_feats = self.node_proj(node_feats)  # [B, T, 2L, gcn_in]

        # --- Temporal GCN ---
        gcn_out = self.temporal_gcn(
            node_feats, adj_mats,
        )  # [B, T, 2L, gcn_out_dim]

        # Take last time step
        gcn_last = gcn_out[:, -1, :, :]  # [B, 2L, gcn_out_dim]

        # --- Attention pool over nodes ---
        attn_logits = self.attn_query(gcn_last).squeeze(-1)  # [B, 2L]
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, 2L]
        graph_emb = torch.einsum(
            "bn,bnd->bd", attn_weights, gcn_last,
        )  # [B, gcn_out_dim]

        # --- Imbalance features ---
        bid_v_last = bid_data[:, -1, :, 1]  # [B, L]
        ask_v_last = ask_data[:, -1, :, 1]  # [B, L]
        imbalance = self._compute_imbalance(bid_v_last, ask_v_last)  # [B, 4]
        imb_emb = self.imbalance_mlp(imbalance)  # [B, 32]

        # --- Combine and project ---
        combined = torch.cat([graph_emb, imb_emb], dim=-1)  # [B, gcn_out+32]
        return self.output_proj(combined)  # [B, output_dim]
