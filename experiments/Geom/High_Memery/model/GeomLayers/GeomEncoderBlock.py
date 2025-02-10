# model/GeomLayers/GeomEncoderBlock.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GeomLayers.DenseGGNN import DenseGGNN
from model.GeomLayers.NodeAggregatorRes import NodeAggregatorRes

class GeomEncoderBlock(nn.Module):
    def __init__(self, d_model: int, in_nodes: int, out_nodes: int, num_ggnn_layers: int = 1):
        super().__init__()
        # DenseGGNN 的 out_channels= d_model=256
        self.mpnn = DenseGGNN(out_channels=d_model, num_layers=num_ggnn_layers)
        self.norm1 = nn.LayerNorm(d_model)

        self.aggregator = NodeAggregatorRes(d_model, in_nodes, out_nodes)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, adj, mask):
        # x:   [B, N, d_model]
        # adj: [B, N, N]
        # mask:[B, N]
        mpnn_out = self.mpnn(x, adj, mask)
        x2 = x + mpnn_out
        x2 = self.norm1(x2)
        x2 = F.relu(x2)

        pfeat, padj, pmask = self.aggregator(x2, adj, mask)
        pfeat = self.norm2(pfeat)
        pfeat = F.relu(pfeat)
        return pfeat, padj, pmask