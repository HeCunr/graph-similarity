# model/GeomLayers/DenseGGNN.py
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
            x: torch.Tensor,    # [B, N, D]
            adj: torch.Tensor,  # [B, N, N]
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

        # 1) 将每个图的 dense 邻接矩阵转换为 (edge_index, edge_weight) 稀疏格式
        edge_indices = []
        for i in range(batch_size):
            # dense_to_sparse会返回 (edge_index, edge_weight)
            e_idx, _ = dense_to_sparse(adj[i])   # e_idx: [2, E]
            # 如果该图没有边， e_idx.shape[1] == 0
            # 无需特殊处理，直接加偏移或不加偏移都可以
            if e_idx.numel() > 0:  # 有边时再做批内偏移
                e_idx = e_idx + i * num_nodes
            edge_indices.append(e_idx)

        # 2) 将所有图的 edge_index 拼接到一起
        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)  # 形状 [2, sum_of_E]
        else:
            # 理论上 edge_indices 不会为空，但如需更安全，可做如下处理
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        # 3) 将节点特征展平为 [B*N, D] 以适配 PYG 的 GatedGraphConv
        x = x.reshape(batch_size * num_nodes, in_channels)

        # 4) 进行 GNN 传播；对空图(无边)的情况，GatedGraphConv不会进行消息传递，但仍会保留原有特征
        output = self.gnn(x, edge_index)  # [B*N, out_channels]

        # 5) Reshape 回原始批次形状 [B, N, out_channels]
        output = output.reshape(batch_size, num_nodes, self.out_channels)

        # 6) 如果提供了 mask，就对无效节点做屏蔽
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
        可选的辅助函数：将批量的 dense adjacency 转为稀疏 edge_index, edge_weight。
        """
        edge_indices = []
        edge_weights = []

        for i in range(batch_size):
            edge_index, weight = dense_to_sparse(adj[i])
            # 偏移 batch
            if edge_index.size(1) > 0:
                edge_index = edge_index + i * num_nodes
            edge_indices.append(edge_index)
            edge_weights.append(weight)

        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_weight = torch.cat(edge_weights, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=adj.device)
            edge_weight = torch.empty((0,), dtype=adj.dtype, device=adj.device)

        return edge_index, edge_weight
