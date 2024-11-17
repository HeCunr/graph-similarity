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
            # 计算池化窗口大小
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
        # 计算每个阶段的目标长度，使用对数尺度
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

        # 转换为卷积格式
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)

        for pool, norm, conv in zip(self.pooling_layers, self.norm_layers, self.conv_layers):
            # 应用池化
            x = pool(x)

            # 应用卷积和归一化
            identity = x
            x = conv(x)
            x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
            x = norm(x)
            x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)

            # 残差连接
            x = x + identity

        # 转换回原始格式
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)

class Bottleneck(nn.Module):
    def __init__(self, d_model, dim_z):
        super().__init__()
        self.linear = nn.Linear(d_model, dim_z)

    def forward(self, x):
        return self.linear(x)

class ProjectionHead(nn.Module):
    def __init__(self, dim_z, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.linear_proj = nn.ModuleList([nn.Linear(dim_z, dim_z) for _ in range(self.n_layers)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers - 1)] + [nn.Identity()])

    def forward(self, z):
        for i in range(self.n_layers):
            z = self.activations[i](self.linear_proj[i](z))
        return z




class DXFTransformer(nn.Module):
    def __init__(self, d_model=256, num_layers=4, dim_z=256, nhead=8,
                 dim_feedforward=512, dropout=0.1, latent_dropout=0.1):
        super().__init__()
        self.embedding = DXFEmbedding(d_model, max_len=4096)
        self.progressive_pool = ProgressivePooling(
            input_length=4096,
            output_length=64,
            d_model=d_model
        )

        self.encoder = TransformerEncoder(d_model, num_layers, nhead, dim_feedforward, dropout)
        self.bottleneck = Bottleneck(d_model, dim_z)
        self.projection = ProjectionHead(dim_z)
        self.dropout = nn.Dropout(latent_dropout)
        self.tanh = nn.Tanh()

    def forward(self, entity_type, entity_params):
        # 嵌入层
        src = self.embedding(entity_type, entity_params)  # [batch_size, seq_len, d_model]

        # 渐进式下采样
        src = self.progressive_pool(src)  # [batch_size, seq_len/64, d_model]

        # 正确的转置
        src = src.permute(1, 0, 2)  # [seq_len/64, batch_size, d_model]

        # 调整padding mask维度
        padding_mask = F.max_pool1d(
            (entity_type == 10).float().unsqueeze(1),
            kernel_size=4096//64,
            stride=4096//64
        ).squeeze(1).bool()

        memory = self.encoder(src, src_key_padding_mask=padding_mask)

        # 计算潜在向量
        padding_mask_float = (~padding_mask).float().unsqueeze(-1)
        padding_mask_float = padding_mask_float.transpose(0, 1)

        z = (memory * padding_mask_float).sum(dim=0) / padding_mask_float.sum(dim=0)
        _z = self.bottleneck(z)

        # 投影和对比学习部分
        z = self.projection(_z)
        proj_z1 = self.dropout(z)
        proj_z2 = self.dropout(z)
        z = self.tanh(z)

        return {
            "proj_z1": proj_z1,
            "proj_z2": proj_z2,
            "representation": _z
        }