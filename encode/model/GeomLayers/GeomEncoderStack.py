# model/GeomLayers/GeomEncoderStack.py

import torch
import torch.nn as nn

from model.GeomLayers.GeomEncoderBlock import GeomEncoderBlock

class GeomEncoderStack(nn.Module):
    """
    多层堆叠的“(MPNN + Aggregator)模块”，将节点数逐层减少到64。
    假设输入节点固定4096。
    """
    def __init__(self, d_model=32):
        super().__init__()
        self.blocks = nn.ModuleList([
            GeomEncoderBlock(d_model, in_nodes=4096, out_nodes=1024, num_ggnn_layers=3),
            GeomEncoderBlock(d_model, in_nodes=1024, out_nodes=256, num_ggnn_layers=3),
            GeomEncoderBlock(d_model, in_nodes=256, out_nodes=64, num_ggnn_layers=3),
        ])

    def forward(self, x, adj, mask):
        """
        x:   [B, 4096, d_model]
        adj: [B, 4096, 4096]
        mask:[B, 4096]
        """
        for block in self.blocks:
            x, adj, mask = block(x, adj, mask)
        return x, adj, mask  # 最终 x: [B, 64, d_model]
