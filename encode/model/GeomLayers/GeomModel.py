# model/GeomLayers/GeomModel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GeomLayers.GeomEncoderStack import GeomEncoderStack
from utils.Geom_position import Sin2DPositionalEncoding

class GeomModel(nn.Module):
    def __init__(self, init_dim=44, d_model=32):
        super().__init__()
        self.d_model = d_model
        # 把初始44维特征投影到32维
        self.linear_proj = nn.Linear(init_dim, d_model)
        # 2D正弦位置编码
        self.pos_encoder = Sin2DPositionalEncoding(d_pos=d_model, max_val=256.0)
        # 堆叠多层: MPNN + aggregator
        self.encoder_stack = GeomEncoderStack(d_model)

    def forward(self, features, pos2d, adj, mask):
        """
        features: [B, N, 44]
        pos2d:    [B, N, 2]
        adj:      [B, N, N]
        mask:     [B, N]
        return:   [B, 64, d_model]
        """
        # 1. 投影features到32
        feat_32 = self.linear_proj(features)  # [B, N, 32]
        # 2. pos2d -> sinusoidal => [B, N, 32]
        pos_32 = self.pos_encoder(pos2d)      # [B, N, 32]
        # 3. 融合 (简单相加)
        x = feat_32 + pos_32  # [B, N, 32]
        # 4. 送入EncoderStack
        x_enc, adj_enc, mask_enc = self.encoder_stack(x, adj, mask)  # [B, 64, 32], [B,64,64], [B,64]
        return x_enc, adj_enc, mask_enc
