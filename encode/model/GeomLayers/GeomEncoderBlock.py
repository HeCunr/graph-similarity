# model/GeomLayers/GeomEncoderBlock.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GeomLayers.DenseGGNN import DenseGGNN
from model.GeomLayers.NodeAggregatorRes import NodeAggregatorRes

class GeomEncoderBlock(nn.Module):
    """
    单个“MPNN + Add&Norm + (Res聚合) + Add&Norm”模块
    N -> K
    """
    def __init__(self, d_model: int, in_nodes: int, out_nodes: int, num_ggnn_layers: int = 3):
        super().__init__()
        self.mpnn = DenseGGNN(out_channels=d_model, num_layers=num_ggnn_layers)
        self.norm1 = nn.LayerNorm(d_model)

        self.aggregator = NodeAggregatorRes(d_model, in_nodes, out_nodes)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, adj, mask):
        """
        x:   [B, N, d_model]
        adj: [B, N, N]
        mask:[B, N]
        return: [B,K,d_model], [B,K,K], [B,K]
        """
        # 1) MPNN
        mpnn_out = self.mpnn(x, adj, mask)  # [B,N,d_model]
        x2 = x + mpnn_out
        x2 = self.norm1(x2)                # [B,N,d_model]
        x2 = F.relu(x2)

        # 2) 聚合(Res)  => [B,K,d_model]
        pfeat, padj, pmask = self.aggregator(x2, adj, mask)
        # 再 LN
        pfeat = self.norm2(pfeat)
        pfeat = F.relu(pfeat)
        return pfeat, padj, pmask
