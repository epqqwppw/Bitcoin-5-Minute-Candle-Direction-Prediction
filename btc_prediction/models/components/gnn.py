"""Graph Neural Network components for order-book modelling.

Provides graph convolution layers, a temporal graph convolution module,
and a utility class that converts raw order-book data into a graph
representation suitable for GNN processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Graph convolution
# ---------------------------------------------------------------------------


class GraphConvLayer(nn.Module):
    """Single graph convolution layer with message passing.

    Implements the operation::

        H' = σ(D⁻¹ A H W)

    where *A* is the adjacency matrix (optionally with self-loops), *D* is its
    degree matrix, *W* is a learnable weight, and *σ* is ReLU.

    Args:
        in_features: Input feature dimension per node.
        out_features: Output feature dimension per node.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single graph convolution step.

        Args:
            node_features: Node feature matrix of shape
                ``[batch, num_nodes, in_features]``.
            adjacency_matrix: Adjacency matrix of shape
                ``[batch, num_nodes, num_nodes]`` (or ``[num_nodes, num_nodes]``
                when shared across the batch).

        Returns:
            Updated node features of shape ``[batch, num_nodes, out_features]``.
        """
        # Add self-loops: A_hat = A + I
        if adjacency_matrix.dim() == 2:
            eye = torch.eye(
                adjacency_matrix.size(0),
                device=adjacency_matrix.device,
                dtype=adjacency_matrix.dtype,
            )
            a_hat = adjacency_matrix + eye  # [N, N]
        else:
            eye = torch.eye(
                adjacency_matrix.size(-1),
                device=adjacency_matrix.device,
                dtype=adjacency_matrix.dtype,
            ).unsqueeze(0)
            a_hat = adjacency_matrix + eye  # [batch, N, N]

        # Row-normalise: D⁻¹ A_hat
        degree = a_hat.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # avoid /0
        a_norm = a_hat / degree  # [batch?, N, N]

        # Message passing: H' = ReLU(D⁻¹ A_hat H W + b)
        support = node_features @ self.weight  # [batch, N, out_features]
        out = a_norm @ support + self.bias     # [batch, N, out_features]
        return torch.relu(out)


# ---------------------------------------------------------------------------
# Temporal graph convolution
# ---------------------------------------------------------------------------


class TemporalGraphConv(nn.Module):
    """Graph convolution combined with temporal (1-D) convolution.

    Processes a *sequence* of graph snapshots: at each time step the GNN
    aggregates spatial information, and a subsequent 1-D convolution captures
    temporal dynamics across steps.

    Args:
        in_channels: Input feature dimension per node.
        out_channels: Output feature dimension per node.
        num_layers: Number of stacked graph convolution layers per time step.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        # Stack of graph convolution layers
        gcn_layers: list[GraphConvLayer] = []
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels
            gcn_layers.append(GraphConvLayer(in_dim, out_channels))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Temporal convolution across the time axis (kernel_size=3, causal)
        self.temporal_conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=2,
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        nn.init.kaiming_normal_(self.temporal_conv.weight, nonlinearity="relu")

    def forward(
        self,
        node_features_sequence: torch.Tensor,
        adjacency_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal graph embeddings.

        Args:
            node_features_sequence: Node features over time, shape
                ``[batch, time, num_nodes, in_channels]``.
            adjacency_matrices: Adjacency matrices, shape
                ``[batch, time, num_nodes, num_nodes]``.

        Returns:
            Temporal graph embeddings of shape
            ``[batch, time, num_nodes, out_channels]``.
        """
        batch, time, num_nodes, _ = node_features_sequence.shape

        # --- spatial: apply GCN at each time step ---
        h = node_features_sequence  # [B, T, N, C_in]
        for gcn in self.gcn_layers:
            # Reshape to process all (batch × time) graphs at once
            h_flat = h.reshape(batch * time, num_nodes, -1)
            a_flat = adjacency_matrices.reshape(batch * time, num_nodes, num_nodes)
            h_flat = gcn(h_flat, a_flat)  # [B*T, N, C_out]
            h = h_flat.reshape(batch, time, num_nodes, -1)
        # h: [B, T, N, out_channels]

        # --- temporal: 1-D conv over the time dimension per node ---
        # Reshape to [B*N, out_channels, T] for Conv1d
        h_perm = h.permute(0, 2, 3, 1).reshape(
            batch * num_nodes, -1, time,
        )  # [B*N, C_out, T]
        h_perm = self.temporal_conv(h_perm)
        h_perm = h_perm[:, :, :time]  # trim causal padding → [B*N, C_out, T]

        # Reshape back to [B, T, N, C_out]
        h_out = h_perm.reshape(batch, num_nodes, -1, time).permute(0, 3, 1, 2)
        return self.layer_norm(h_out)  # [batch, time, num_nodes, out_channels]


# ---------------------------------------------------------------------------
# Order-book → graph builder
# ---------------------------------------------------------------------------


class OrderBookGraphBuilder:
    """Converts raw order-book bid/ask data into a graph representation.

    Each price level becomes a node with features ``[price, volume, side]``
    (side: 0 = bid, 1 = ask).  Edges are weighted by volume similarity and
    price proximity.

    Args:
        num_levels: Number of price levels on each side of the book.
    """

    def __init__(self, num_levels: int = 10) -> None:
        self.num_levels = num_levels

    def build_graph(
        self,
        bid_prices: torch.Tensor,
        bid_volumes: torch.Tensor,
        ask_prices: torch.Tensor,
        ask_volumes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a graph from one snapshot of the order book.

        Args:
            bid_prices: Bid prices, shape ``[batch, num_levels]``.
            bid_volumes: Bid volumes, shape ``[batch, num_levels]``.
            ask_prices: Ask prices, shape ``[batch, num_levels]``.
            ask_volumes: Ask volumes, shape ``[batch, num_levels]``.

        Returns:
            Tuple of:
            - node_features: ``[batch, 2 * num_levels, 3]``
              (price, volume, side).
            - adjacency_matrix: ``[batch, 2 * num_levels, 2 * num_levels]``.
        """
        batch = bid_prices.size(0)
        num_nodes = 2 * self.num_levels
        device = bid_prices.device

        # --- build node features ---
        bid_side = torch.zeros(batch, self.num_levels, 1, device=device)
        ask_side = torch.ones(batch, self.num_levels, 1, device=device)

        bid_features = torch.stack([bid_prices, bid_volumes], dim=-1)
        ask_features = torch.stack([ask_prices, ask_volumes], dim=-1)

        bid_nodes = torch.cat([bid_features, bid_side], dim=-1)  # [B, L, 3]
        ask_nodes = torch.cat([ask_features, ask_side], dim=-1)  # [B, L, 3]

        node_features = torch.cat([bid_nodes, ask_nodes], dim=1)  # [B, 2L, 3]

        # --- build adjacency matrix ---
        prices = node_features[:, :, 0]    # [B, 2L]
        volumes = node_features[:, :, 1]   # [B, 2L]

        # Price proximity: exp(-|p_i - p_j| / σ_p)
        price_diff = (
            prices.unsqueeze(2) - prices.unsqueeze(1)
        ).abs()  # [B, 2L, 2L]
        price_std = prices.std(dim=1).unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
        price_sim = torch.exp(-price_diff / price_std)

        # Volume similarity: exp(-|v_i - v_j| / σ_v)
        vol_diff = (
            volumes.unsqueeze(2) - volumes.unsqueeze(1)
        ).abs()  # [B, 2L, 2L]
        vol_std = volumes.std(dim=1).unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
        vol_sim = torch.exp(-vol_diff / vol_std)

        # Combined edge weights (element-wise product)
        adjacency_matrix = price_sim * vol_sim  # [B, 2L, 2L]

        # Zero out self-loops (they are added inside GraphConvLayer)
        eye = torch.eye(num_nodes, device=device).unsqueeze(0)
        adjacency_matrix = adjacency_matrix * (1.0 - eye)

        return node_features, adjacency_matrix
