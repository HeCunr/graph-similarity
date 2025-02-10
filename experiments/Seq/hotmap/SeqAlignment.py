# model/SeqLayers/SeqAlignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SeqLayers.seq_transformer_encoder import TwoLayerMLP

class NodeAlignmentHead(nn.Module):
    """
    用于跨视图的节点对齐 + 最终投影
    """

    def __init__(self, d_model: int, alignment='concat',  latent_dropout=0.1):
        super().__init__()
        self.alignment = alignment
        # 1) 对齐层: concat 后 => Linear(d_model*2 -> d_model)
        if alignment.lower() == 'concat':
            self.alignment_layer = nn.Linear(2 * d_model, d_model)
        elif alignment.lower() == 'bilinear':
            raise NotImplementedError("bilinear alignment not implemented.")
        else:
            raise NotImplementedError(f"Unknown alignment={alignment}")

        # 2) 最终投影 (之前在 SeqTransformer 里的 projection)
        self.projection = TwoLayerMLP(hidden_dim=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(latent_dropout)

    def perform_alignment(self, z_view1: torch.Tensor, z_view2: torch.Tensor):
        """
        z_view1,z_view2: [B,N,d_model]
        Step1: 余弦注意力 => 双向加权
        Step2: concat => alignment_layer => ...
        Step3: 这里新增投影 projection + BN + Dropout
        return: out1, out2 => [B,N,d_model] (对齐 + 投影后)
        """
        # (1) 计算节点对齐注意力 = cos(v1_i, v2_j)
        attention = self.node_alignment_attention(z_view1, z_view2)  # [B,N,N]

        # (2) 用 attention 加权 v2 => v1; v1 => v2
        att_v2 = torch.bmm(attention, z_view2)  # [B,N,d_model]
        att_v1 = torch.bmm(attention.transpose(1, 2), z_view1)  # [B,N,d_model]

        # (3) 融合
        if self.alignment.lower() == 'concat':
            merged1 = torch.cat([z_view1, att_v2], dim=-1)  # [B,N,2d]
            merged2 = torch.cat([z_view2, att_v1], dim=-1)  # [B,N,2d]
            out1 = self.alignment_layer(merged1)  # => [B,N,d_model]
            out2 = self.alignment_layer(merged2)
        else:
            # 如果有其他策略，如 'bilinear'，可自行添加
            raise NotImplementedError()

        # (4) 现在再做最终投影(MLP + BN + Dropout)
        B, N, d = out1.shape
        out1_flat = out1.reshape(B * N, d)
        out2_flat = out2.reshape(B * N, d)

        out1_flat = self.projection(out1_flat)  # => (B*N, d)
        out1_flat = self.bn(out1_flat)  # BatchNorm1d: 需要输入 (batch, features)
        out1_flat = self.dropout(out1_flat)

        out2_flat = self.projection(out2_flat)
        out2_flat = self.bn(out2_flat)
        out2_flat = self.dropout(out2_flat)

        # reshape 回 [B,N,d]
        out1 = out1_flat.view(B, N, d)
        out2 = out2_flat.view(B, N, d)

        return out1, out2

    def node_alignment_attention(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise node alignment attention [B,N,N] = cos(v1_i, v2_j)."""
        v1_norm = F.normalize(v1, dim=-1)
        v2_norm = F.normalize(v2, dim=-1)
        att = torch.bmm(v1_norm, v2_norm.transpose(1, 2))  # [B,N,N]
        return att
