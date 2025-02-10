# model/GeomLayers/SparseGeomEncoderBlock.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GeomLayers.SparseMPNN import SparseMPNN
from model.GeomLayers.SparseNodeAggregator import SparseNodeAggregator

class SparseGeomEncoderBlock(nn.Module):
    def __init__(self, d_model: int, in_nodes: int, out_nodes: int, num_layers: int = 1):
        super().__init__()
        # 稀疏MPNN
        self.mpnn = SparseMPNN(out_channels=d_model, num_layers=num_layers)
        self.norm1 = nn.LayerNorm(d_model)

        self.aggregator = SparseNodeAggregator(
            in_features=d_model,
            in_nodes=in_nodes,
            out_nodes=out_nodes
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_index_list, edge_weight_list, mask):
        """
        x: [B, N, d_model]
        edge_index_list:  list of length B, each shape [2, E]
        edge_weight_list: list of length B, each shape [E]
        mask: [B, N]
        return:
          pfeat: [B, out_nodes, d_model]
          p_edge_index_list: new edges
          p_edge_weight_list: new edge weights
          p_mask: [B, out_nodes]
        """
        mpnn_out = self.mpnn(x, edge_index_list, edge_weight_list, mask)
        x2 = x + mpnn_out
        x2 = self.norm1(x2)
        x2 = F.relu(x2)

        pfeat, p_edge_index_list, p_edge_weight_list, pmask = self.aggregator(
            x2, edge_index_list, edge_weight_list, mask
        )
        pfeat = self.norm2(pfeat)
        pfeat = F.relu(pfeat)

        return pfeat, p_edge_index_list, p_edge_weight_list, pmask
