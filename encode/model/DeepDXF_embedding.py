
import torch
import torch.nn as nn

class PositionalEncodingLUT(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_embed(positions)
        return self.dropout(x)

class DXFEmbedding(nn.Module):
    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.entity_embed = nn.Embedding(11, d_model)  # 11 entity types including EOS
        self.param_embed1 = nn.Embedding(257, 64)  # 256 parameter values + 1 for padding (-1)
        self.param_embed2 = nn.Linear(64 * 38, d_model)  # 38 parameters per entity
        self.pos_encoding = PositionalEncodingLUT(d_model, max_len=max_len)

    def forward(self, entity_type, entity_params):
        batch_size, sequence_length = entity_type.shape

        entity_type_embed = self.entity_embed(entity_type)

        entity_params_embed = self.param_embed1(entity_params.long() + 1)  # Shift by 1 to handle -1
        entity_params_embed = entity_params_embed.view(batch_size, sequence_length, -1)
        entity_params_embed = self.param_embed2(entity_params_embed)

        entity_embed = entity_type_embed + entity_params_embed
        entity_embed = self.pos_encoding(entity_embed)

        return entity_embed

