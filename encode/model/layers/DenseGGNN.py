import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_geometric.utils import dense_to_sparse

class DenseGGNN(nn.Module):
    """
    Dense implementation of Gated Graph Neural Networks.
    Handles batched dense adjacency matrices by converting to sparse format internally.
    """

    def __init__(self, out_channels: int, num_layers: int = 1):
        """
        Initialize DenseGGNN layer.

        Args:
            out_channels (int): Number of output channels
            num_layers (int): Number of GNN layers to apply
        """
        super(DenseGGNN, self).__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.gnn = GatedGraphConv(
            out_channels=out_channels,
            num_layers=num_layers,
            aggr='add',  # Use additive aggregation
            bias=True
        )

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of DenseGGNN.

        Args:
            x (torch.Tensor): Node features tensor of shape [B, N, D]
            adj (torch.Tensor): Adjacency matrix tensor of shape [B, N, N]
            mask (torch.Tensor, optional): Mask tensor of shape [B, N]

        Returns:
            torch.Tensor: Updated node features of shape [B, N, out_channels]
        """
        batch_size, num_nodes, in_channels = x.size()

        # Convert to sparse format for each graph in batch
        edge_indices = []
        for i in range(batch_size):
            # Get sparse indices for current graph
            edge_index = dense_to_sparse(adj[i])[0]
            # Offset node indices by batch index
            edge_indices.append(edge_index + i * num_nodes)

        # Concatenate all edge indices
        edge_index = torch.cat(edge_indices, dim=1)

        # Reshape node features for sparse format
        x = x.reshape(-1, in_channels)

        # Apply GNN
        output = self.gnn(x, edge_index)

        # Reshape back to dense format
        output = output.reshape(batch_size, num_nodes, self.out_channels)

        # Apply mask if provided
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        return output

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        self.gnn.reset_parameters()

    @staticmethod
    def get_sparse_adj(
            adj: torch.Tensor,
            batch_size: int,
            num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert batched dense adjacency matrices to sparse format.

        Args:
            adj (torch.Tensor): Dense adjacency tensor [B, N, N]
            batch_size (int): Batch size
            num_nodes (int): Number of nodes per graph

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices and edge weights
        """
        edge_indices = []
        edge_weights = []

        for i in range(batch_size):
            # Get sparse representation
            edge_index, weight = dense_to_sparse(adj[i])
            # Offset node indices
            edge_indices.append(edge_index + i * num_nodes)
            edge_weights.append(weight)

        edge_index = torch.cat(edge_indices, dim=1)
        edge_weight = torch.cat(edge_weights, dim=0)

        return edge_index, edge_weight