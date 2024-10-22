import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DeepDXF_embedding import DXFEmbedding

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
        self.layers = nn.ModuleList([nn.Linear(dim_z, dim_z) for _ in range(n_layers)])
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

class DXFTransformer(nn.Module):
    def __init__(self, d_model=256, num_layers=4, dim_z=256, nhead=8, dim_feedforward=512, dropout=0.1, latent_dropout=0.1):
        super().__init__()
        self.embedding = DXFEmbedding(d_model)
        self.encoder = TransformerEncoder(d_model, num_layers, nhead, dim_feedforward, dropout)
        self.bottleneck = Bottleneck(d_model, dim_z)
        self.projection = ProjectionHead(dim_z)
        self.latent_dropout = nn.Dropout(latent_dropout)

    def forward(self, entity_type, entity_params):
        print(f"Input entity_type shape: {entity_type.shape}")
        print(f"Input entity_params shape: {entity_params.shape}")
        assert entity_type.dtype == torch.long, f"Expected entity_type to be Long, but got {entity_type.dtype}"
        assert entity_params.dtype == torch.float, f"Expected entity_params to be Float, but got {entity_params.dtype}"
        src = self.embedding(entity_type, entity_params)
        src = src.transpose(0, 1)  # (N, S, E) -> (S, N, E)

        padding_mask = (entity_type == 10)  # Assuming 10 is the index for EOS
        memory = self.encoder(src, src_key_padding_mask=padding_mask)

        # Convert padding_mask to float and use it for averaging
        padding_mask_float = (~padding_mask).float().unsqueeze(-1)  # (N, S, 1)
        padding_mask_float = padding_mask_float.transpose(0, 1)  # (S, N, 1)

        print(f"memory shape: {memory.shape}")
        print(f"padding_mask_float shape: {padding_mask_float.shape}")

        z = (memory * padding_mask_float).sum(dim=0) / padding_mask_float.sum(dim=0)

        z = self.bottleneck(z)
        proj_z = self.projection(z)

        # Apply dropout twice to get two augmented versions
        proj_z1 = self.latent_dropout(proj_z)
        proj_z2 = self.latent_dropout(proj_z)

        return z, proj_z1, proj_z2
