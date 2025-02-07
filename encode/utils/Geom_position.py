# utils/Geom_position.py
import torch
import torch.nn as nn
import numpy as np

class Sin2DPositionalEncoding(nn.Module):
    """
    2D 正弦位置编码，将 (x, y) in [0,255] 的坐标映射到 d_pos 维向量。
    假设 d_pos 为 32，则前 16 维编码 x，后 16 维编码 y。
    """
    def __init__(self, d_pos=32, max_val=256.0):
        super().__init__()
        self.d_pos = d_pos
        self.max_val = max_val
        # 将 d_pos 一分为二，用于 x,y 分别做正弦编码
        self.d_half = d_pos // 2
        assert self.d_half * 2 == d_pos, "d_pos必须是偶数"

    def forward(self, pos2d: torch.Tensor) -> torch.Tensor:
        """
        pos2d: [B, N, 2], 值范围 ~ [0,255]
        return: [B, N, d_pos]
        """
        # 分离 x, y
        x = pos2d[..., 0]  # [B, N]
        y = pos2d[..., 1]  # [B, N]

        # 归一化 (可选): 除以 self.max_val => ~ [0,1]
        x = x / self.max_val
        y = y / self.max_val

        # 构建PE
        pe_x = self._sinusoidal_encoding_1d(x, self.d_half)  # [B, N, d_half]
        pe_y = self._sinusoidal_encoding_1d(y, self.d_half)  # [B, N, d_half]

        return torch.cat([pe_x, pe_y], dim=-1)  # [B, N, d_pos]

    def _sinusoidal_encoding_1d(self, coords, d_embed):
        """
        coords: [B,N], 取值 ~ [0,1]
        d_embed: 一半维度
        return: [B, N, d_embed]
        """
        device = coords.device
        B, N = coords.shape

        # 位置编码公式： pos / (10000^(2i/d_embed))
        # 先把 coords.view(-1,1) => shape [B*N,1]
        pos = coords.view(-1, 1)  # [B*N,1]
        dim_idx = torch.arange(d_embed, dtype=torch.float, device=device).view(1, -1)  # [1,d_embed]

        # log_scale =  2i / d_embed
        # denominator = 10000^(log_scale)
        # shape [B*N, d_embed]
        denom = torch.exp(dim_idx * -(np.log(10000.0) / d_embed))

        # [B*N, d_embed]
        scaled_pos = pos * denom
        # 偶数维用 sin，奇数维用 cos
        sin_mask = torch.arange(d_embed, device=device).view(1, -1) % 2 == 0
        sin_mask = sin_mask.float()  # True->1, False->0
        cos_mask = 1.0 - sin_mask

        encoded = torch.sin(scaled_pos) * sin_mask + torch.cos(scaled_pos) * cos_mask
        # reshape 回 [B, N, d_embed]
        return encoded.view(B, N, d_embed)

def drop_pos2d(pos2d: torch.Tensor, drop_prob: float):
    """
    对 pos2d 做简单的“dropout”，随机将部分坐标置为 0。
    pos2d: [B, N, 2]
    drop_prob: 0 ~ 1
    """
    B, N, _ = pos2d.shape
    # 这里的做法类似特征dropout: 每个sample中的若干坐标被置0
    for i in range(B):
        drop_mask = torch.empty((N,), device=pos2d.device).uniform_(0,1) < drop_prob
        pos2d[i, drop_mask, :] = 0.0
    return pos2d
