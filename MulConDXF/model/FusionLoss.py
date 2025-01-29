# model/FusionLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeomSeqClipLoss(nn.Module):
    """
    对 B 个 (geom_repr, seq_repr) 做对比学习，类似 CLIP:
    logits_{ij} = (g_i \cdot s_j) / temperature
    i=j 为正样本
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, geom_repr, seq_repr):
        """
        geom_repr: [B, d], 已 l2_normalize
        seq_repr:  [B, d], 也已 l2_normalize
        返回标量 loss
        """
        # 余弦相似 / temperature
        logits = torch.matmul(geom_repr, seq_repr.t())  # [B,B]
        logits = logits / self.temperature

        labels = torch.arange(geom_repr.size(0), dtype=torch.long, device=geom_repr.device)
        loss_i = F.cross_entropy(logits,     labels)  # geom->seq
        loss_j = F.cross_entropy(logits.t(), labels)  # seq->geom
        return 0.5*(loss_i + loss_j)
