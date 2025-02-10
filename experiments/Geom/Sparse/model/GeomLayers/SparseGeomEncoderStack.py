# model/GeomLayers/SparseGeomEncoderStack.py

import torch
import torch.nn as nn

from model.GeomLayers.SparseGeomEncoderBlock import SparseGeomEncoderBlock
from model.GeomLayers.SparseGGNNBlock import SparseGGNNBlock

class SparseGeomEncoderStack(nn.Module):
    """
    稀疏版的多层堆叠:
      1) (SparseMPNN + SparseNodeAggregator) 3次
      2) SparseGGNNBlock (可多层)
    例如节点从 4096 -> 1024 -> 256 -> 64，再在64节点上跑GNN。
    """

    def __init__(self, args, d_model=256):
        super().__init__()
        self.blocks = nn.ModuleList([
            SparseGeomEncoderBlock(d_model, in_nodes=4096, out_nodes=1024),
            SparseGeomEncoderBlock(d_model, in_nodes=1024, out_nodes=256),
            SparseGeomEncoderBlock(d_model, in_nodes=256, out_nodes=64),
            SparseGGNNBlock(node_init_dims=d_model, args=args)
        ])

    def forward(self, x, edge_index_list, edge_weight_list, mask):
        """
        x:              [B, N, d_model]，初始 N=4096
        edge_index_list:[B个] [2, E]  的列表
        edge_weight_list:[B个] [E]    的列表
        mask:           [B, N]
        return:
          x_out: [B, N', d_model]，最后一层的输出(示例N'=64)
        """
        for block in self.blocks:
            if isinstance(block, SparseGGNNBlock):
                # 只更新x，不改动节点数
                x = block(x, edge_index_list, edge_weight_list, mask)
            else:
                # 同时更新 x, edge_index_list, edge_weight_list, mask
                x, edge_index_list, edge_weight_list, mask = block(
                    x, edge_index_list, edge_weight_list, mask
                )
        return x
