# model/SeqLayers/SelfAttPool.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttPool(nn.Module):
    """
    A simple self-attention pooling that maps [B,S,d] -> [B,d].
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        x: [B,S,d]
        mask: [B,S], 1=valid, 0=invalid
        return: [B,d]
        """
        B, S, d = x.shape
        Q = self.query_proj(x)  # [B,S,d]
        K = self.key_proj(x)
        V = self.value_proj(x)

        scores = torch.bmm(Q, K.transpose(1,2)) / (d**0.5)  # [B,S,S]

        if mask is not None:
            # mask == 0 => -inf
            mask_ = (mask==0).unsqueeze(1).expand(B, S, S)
            scores = scores.masked_fill(mask_, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # [B,S,S]
        out  = torch.bmm(attn, V)        # [B,S,d]
        pooled = out.mean(dim=1)         # [B,d], 也可以加权avg
        return pooled
