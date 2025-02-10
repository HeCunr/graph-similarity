# model/GeomLayers/SparseGeomModel.py

import torch
import torch.nn as nn

from model.GeomLayers.Geom_embedding import GeomEmbedding
from model.GeomLayers.SparseGeomEncoderStack import SparseGeomEncoderStack


class SparseGeomModel(nn.Module):
    """
    基于 PyG 稀疏 edge_index 的 GeomModel。
    """

    def __init__(self, args, d_model=256):
        super().__init__()
        self.args = args
        self.d_model = d_model

        # 1) Embedding
        self.embedding = GeomEmbedding(d_model=self.d_model)

        # 2) 多层堆叠 (MPNN + 聚合 + 最终GGNN)，稀疏版
        self.encoder_stack = SparseGeomEncoderStack(args, d_model=self.d_model)

    def forward(self, features, pos2d, edge_index_list, edge_weight_list, mask):
        """
        Params:
          features:        [B, N, 44]
          pos2d:           [B, N, 2]
          edge_index_list: List[Tensor], 每个Tensor形状 [2, E] ，长度=B
          edge_weight_list:List[Tensor], 每个Tensor形状 [E]   ，长度=B
          mask:            [B, N]

        Return:
          x_out: [B, N', d_model]，注意N'可能缩减（若Aggregator把节点数缩小）
                 目前例子里最后缩到 64
        """
        # 1) 先embedding
        x = self.embedding(features, pos2d)  # [B, N, d_model]

        # 2) 堆叠编码器
        x_out = self.encoder_stack(x, edge_index_list, edge_weight_list, mask)
        return x_out
