#model/transformer_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DeepDXF_embedding import DXFEmbedding

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
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        x = x.transpose(1, 2)  # => (batch_size, d_model, seq_len)

        for pool, norm, conv in zip(self.pooling_layers, self.norm_layers, self.conv_layers):
            x = pool(x)  # Conv -> GELU -> MaxPool
            identity = x
            x = conv(x)
            x = x.transpose(1, 2)  # => (batch_size, seq_len', d_model)
            x = norm(x)
            x = x.transpose(1, 2)  # => (batch_size, d_model, seq_len')
            x = x + identity  # 残差

        x = x.transpose(1, 2)  # => (batch_size, seq_len', d_model)
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
        # x: [seq_len, batch_size, d_model]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)

##
# 2 层 MLP 投影头（维度保持 256 不变）
# 这里去掉了原有的 Bottleneck
##
class TwoLayerMLP(nn.Module):
    """
    输入输出均是 (batch, hidden_dim=256)，中间隐层同样256，不改变维度
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x shape: (N, 256) 或 (any, 256)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class DXFTransformer(nn.Module):
    """
    修改要点:
      1) 去除原有 Bottleneck，保留embedding, progressive_pool, encoder
      2) 最终不做时序平均, 保持 (batch_size, 64, 256)
      3) 用2层MLP(维度256->256) + BN, 结果仍 (batch_size,64,256)
    """
    def __init__(self,
                 d_model=256,
                 num_layers=6,
                 dim_z=256,       # 此处留作兼容，但不会再特别用
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0.1,
                 latent_dropout=0.1):
        super().__init__()
        self.embedding = DXFEmbedding(d_model, max_len=4096)
        self.progressive_pool = ProgressivePooling(
            input_length=4096,
            output_length=64,
            d_model=d_model
        )
        self.encoder = TransformerEncoder(d_model, num_layers, nhead, dim_feedforward, dropout)

        # ====== 2层MLP投影到对比空间(保持 256 -> 256) ======
        self.projection = TwoLayerMLP(hidden_dim=d_model)
        self.bn = nn.BatchNorm1d(d_model)  # 对最后一维256做BN
        self.dropout = nn.Dropout(latent_dropout)

    def forward(self, entity_type, entity_params):
        """
        输入: entity_type=(2N,4096), entity_params=(2N,4096,43)
        产出: proj_z1, proj_z2 分别是 [N,64,256]
        """
        # 1) 嵌入 => (2N, 4096, 256)
        src = self.embedding(entity_type, entity_params)

        # 2) progressive_pool => (2N, 64, 256)
        src = self.progressive_pool(src)

        # 3) 变成 [seq_len=64, batch=2N, d_model=256] 喂 transformer
        src = src.permute(1, 0, 2)  # => (64, 2N, 256)
        memory = self.encoder(src)  # => (64, 2N, 256)
        memory = memory.permute(1, 0, 2)  # => (2N, 64, 256)

        # 4) 2层MLP投影, + BN, + dropout
        #    先 flatten到 (2N*64,256) 做全连接, BN, 再 reshape回来
        B, S, C = memory.shape  # B=2N, S=64, C=256
        x_flat = memory.reshape(B * S, C)    # => (2N*64,256)
        x_flat = self.projection(x_flat)     # => (2N*64,256)
        x_flat = self.bn(x_flat)             # => (2N*64,256)
        x_flat = self.dropout(x_flat)        # => (2N*64,256)
        memory_proj = x_flat.reshape(B, S, C) # => (2N,64,256)

        # 5) 把前 N 与后 N 分割，得到 proj_z1, proj_z2
        #    proj_z1.shape = (N,64,256), proj_z2.shape=(N,64,256)
        mid = B // 2
        proj_z1 = memory_proj[:mid]
        proj_z2 = memory_proj[mid:]

        return {
            "proj_z1": proj_z1,
            "proj_z2": proj_z2
        }
