# model/seq_transformer_encoder.py
import torch
import torch.nn as nn

from model.SeqLayers.Seq_embedding import SeqEmbedding

class ProgressivePooling(nn.Module):
    def __init__(self, input_length=2048, output_length=64, d_model=256):
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
        # x: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)  # => (batch_size, d_model, seq_len)
        for pool, norm, conv in zip(self.pooling_layers, self.norm_layers, self.conv_layers):
            x = pool(x)  # Conv->GELU->MaxPool
            identity = x
            x = conv(x)
            x = x.transpose(1, 2)  # => (batch_size, seq_len', d_model)
            x = norm(x)
            x = x.transpose(1, 2)
            x = x + identity
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
    修改后:
      - forward仅处理一个批次(B,2048,44)
      - 返回 (B,64,256)，不再拆成proj_z1, proj_z2
    """
    def __init__(self,
                 d_model=256,
                 num_layers=6,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        self.embedding = SeqEmbedding(d_model, max_len=2048)
        self.progressive_pool = ProgressivePooling(
            input_length=2048,
            output_length=64,
            d_model=d_model
        )
        self.encoder = TransformerEncoder(d_model, num_layers, nhead, dim_feedforward, dropout)

    def forward(self, entity_type, entity_params):
        """
        输入:
            entity_type   (B,2048)
            entity_params (B,2048,43)
        输出:
            memory_proj   (B,64,256)
        """
        # 1) 嵌入
        src = self.embedding(entity_type, entity_params)          # => (B, 2048, 256)

        # 2) Progressive Pooling
        src = self.progressive_pool(src)                          # => (B, 64, 256)

        # 3) Transformer 编码: 先转[seq_len, batch, d_model]
        src = src.permute(1, 0, 2)                                # => (64, B, 256)
        memory = self.encoder(src)                                # => (64, B, 256)
        memory = memory.permute(1, 0, 2)                          # => (B, 64, 256)

        return memory
