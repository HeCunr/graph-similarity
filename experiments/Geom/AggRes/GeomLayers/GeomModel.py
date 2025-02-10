#model/GeomLayers/GeomModel.py
import torch
import torch.nn as nn
from model.GeomLayers.GeomEncoderStack import GeomEncoderStack
from model.GeomLayers.Geom_embedding import GeomEmbedding


class GeomModel(nn.Module):
    def __init__(self, args,  d_model=256):
        """
        init_dim = 44 (feature的列数),
        d_model  = 256 (最终的嵌入维度, 与 GGNN/MPNN 对应)
        """
        super().__init__()
        self.d_model = d_model

        # 1) 用 GeomEmbedding 将输入的 (features, pos2d) => [B, N, d_model=256]
        self.embedding = GeomEmbedding(d_model=self.d_model)

        # 2) GeomEncoderStack: 先三层 (MPNN+聚合), 然后再加一个 GeomGGNNBlock
        #   这里把 config 里的所有参数 (filters, conv, dropout 等) 一并传进去
        self.encoder_stack = GeomEncoderStack(args=args, d_model=self.d_model)

    def forward(self, features, pos2d, adj, mask):
        """
        features: [B, N, 44]
        pos2d:    [B, N, 2]
        adj:      [B, N, N]
        mask:     [B, N]

        return:
          x_enc:   [B, 64, d_model=256]
          adj_enc: [B, 64, 64]
          mask_enc:[B, 64]
        """
        # 1) 先通过 embedding 得到 x: [B, N, 256]
        x = self.embedding(features, pos2d)

        # 2) 堆叠编码器
        x_enc, adj_enc, mask_enc = self.encoder_stack(x, adj, mask)
        return x_enc, adj_enc, mask_enc
