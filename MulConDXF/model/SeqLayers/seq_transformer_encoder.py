# model/SeqLayers/seq_transformer_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SeqLayers.Seq_embedding import SeqEmbedding
from model.SeqLayers.SelfAttPool import SelfAttPool

class ProgressivePooling(nn.Module):
    def __init__(self, input_length=4096, output_length=64, d_model=256):
        super().__init__()
        self.stages = self._calculate_stages(input_length, output_length)
        self.pooling_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        current_length = input_length
        for target_length in self.stages:
            pool_size = current_length // target_length
            self.pooling_layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
            ))
            self.norm_layers.append(nn.LayerNorm(d_model))
            self.conv_layers.append(nn.Conv1d(d_model, d_model, kernel_size=3, padding=1))
            current_length = target_length

    def _calculate_stages(self, input_length, output_length):
        stages = []
        current = input_length
        while current > output_length:
            current = current // 2
            if current <= output_length:
                stages.append(output_length)
                break
            stages.append(current)
        return stages

    def forward(self, x):
        # x: [B, seq_len, d_model]
        x = x.transpose(1, 2)  # => [B, d_model, seq_len]
        for pool, norm, conv in zip(self.pooling_layers, self.norm_layers, self.conv_layers):
            x = pool(x)     # Conv->GELU->MaxPool
            identity = x
            x = conv(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            x = x + identity
        x = x.transpose(1, 2)  # => [B, new_seq_len, d_model]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        # x: [seq_len, B, d_model]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)

class TwoLayerMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class SeqTransformer(nn.Module):
    """
    在原先的基础上，额外加入 self_att_pool 用于输出 [B,d]
    给“Geom-Seq CL”使用。
    """
    def __init__(self,
                 d_model=256,
                 num_layers=6,
                 dim_z=256,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0.1,
                 latent_dropout=0.1,
                 use_selfatt_pool=True):
        super().__init__()
        self.embedding = SeqEmbedding(d_model, max_len=4096)
        self.progressive_pool = ProgressivePooling(4096, 64, d_model)
        self.encoder = TransformerEncoder(d_model, num_layers, nhead, dim_feedforward, dropout)

        # 原投影头
        self.projection = TwoLayerMLP(hidden_dim=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(latent_dropout)

        # 新增：Self-attention pooling + 投影，用于输出图级向量
        self.use_selfatt_pool = use_selfatt_pool
        if use_selfatt_pool:
            self.self_att_pool = SelfAttPool(d_model)
            self.fusion_proj   = nn.Linear(d_model, d_model)

    def forward(self, entity_type, entity_params, return_fusion=False):
        """
        Args:
          entity_type:   [B,4096]
          entity_params: [B,4096,43]
          return_fusion: 是否返回 self-att-pool 输出 [B,d]
        Returns:
          memory_proj: [B,64,256]
          (可选) fused_vec: [B,256] for Geom-Seq CL
        """
        # 1) 嵌入 => [B,4096,256]
        src = self.embedding(entity_type, entity_params)

        # 2) 渐进池化 => [B,64,256]
        src = self.progressive_pool(src)

        # 3) Transformer => 仍然输出 [B,64,256]
        src = src.permute(1,0,2)  # => [64, B, 256]
        memory = self.encoder(src)   # => [64, B, 256]
        memory = memory.permute(1,0,2)  # => [B,64,256]

        # 4) 原来的 2层MLP投影 => [B,64,256]
        B, S, C = memory.shape
        x_flat = memory.reshape(B*S, C)
        x_flat = self.projection(x_flat)
        x_flat = self.bn(x_flat)
        x_flat = self.dropout(x_flat)
        memory_proj = x_flat.reshape(B, S, C)

        if self.use_selfatt_pool and return_fusion:
            # (可选) 利用 self-att-pool 获得 [B,256]
            fused_vec = self.self_att_pool(memory)  # => [B,256]
            fused_vec = self.fusion_proj(fused_vec) # => [B,256]
            fused_vec = F.normalize(fused_vec, dim=-1)
            return memory_proj, fused_vec
        else:
            return memory_proj
