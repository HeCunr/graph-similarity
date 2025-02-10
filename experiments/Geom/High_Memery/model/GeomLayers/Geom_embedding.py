#model/GeomLayers/Geom_embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeomEmbedding(nn.Module):
    """
    将 (features, pos2d) 映射到 d_model=256 的向量
    features: [B, N, 44], 其中
      - features[:, :, 0] 是实体类型 t_i ∈ {0..11 或 -1}
      - features[:, :, 1:] 是 43 维实体参数(取值 -1..255)
    pos2d: [B, N, 2], 取值 -1..255
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        # ---------- 1) 实体类型嵌入 ----------
        #   13个索引：0..11 对应实体类型，12 对应 -1 (无效)
        #   输出维度 d_model
        self.embedding_cmd = nn.Embedding(13, d_model)

        # ---------- 2) 实体参数嵌入 ----------
        #   param_dim=43，param_embed_dim=64
        #   257个索引：0..255 对应参数取值，256 对应 -1
        self.param_dim = 43
        self.param_embed_dim = 64
        self.embedding_param_b = nn.Embedding(257, self.param_embed_dim)
        #   把 43 * 64 => 256
        self.param_linear = nn.Linear(self.param_dim * self.param_embed_dim, d_model)

        # ---------- 3) 位置嵌入 ----------
        #   pos_dim=2, pos_embed_dim=64
        #   同样 257个索引：0..255 对应坐标取值，256 对应 -1
        self.pos_dim = 2
        self.pos_embed_dim = 64
        self.embedding_pos_b = nn.Embedding(257, self.pos_embed_dim)
        #   把 2 * 64 => 256
        self.pos_linear = nn.Linear(self.pos_dim * self.pos_embed_dim, d_model)

    def forward(self, features: torch.Tensor, pos2d: torch.Tensor):
        """
        features: [B, N, 44]
        pos2d:    [B, N, 2]
        return:   [B, N, 256]
        """
        # =============== Part A: 实体类型嵌入 ===============
        entity_type = features[:, :, 0]  # [B, N]
        # -1 => 索引 12；其他(0..11)对应相同索引
        entity_type_idx = torch.where(
            entity_type < 0,
            torch.tensor(12, device=entity_type.device),
            entity_type.long()
        )  # [B, N]
        # 直接用 nn.Embedding 查表
        e_cmd = self.embedding_cmd(entity_type_idx)  # => [B, N, d_model]

        # =============== Part B: 实体参数嵌入 ===============
        # param_matrix: [B, N, 43], -1 => 索引 256
        param_matrix = features[:, :, 1:]  # [B, N, 43]
        param_idx = torch.where(
            param_matrix < 0,
            torch.tensor(256, device=param_matrix.device),
            param_matrix.long()
        )  # => [B, N, 43]

        # 先在第3维(每个参数)上用 embedding => [B, N, 43, param_embed_dim]
        e_param_b = self.embedding_param_b(param_idx)
        # 再 flatten => [B, N, 43*64]
        B, N, P, C = e_param_b.shape
        e_param_flat = e_param_b.view(B, N, P * C)
        # 通过线性层 => [B, N, d_model]
        e_param = self.param_linear(e_param_flat)

        # =============== Part C: 位置嵌入 ===============
        # pos2d: [B, N, 2], -1 => 索引 256
        pos_idx = torch.where(
            pos2d < 0,
            torch.tensor(256, device=pos2d.device),
            pos2d.long()
        )  # => [B, N, 2]

        # => [B, N, 2, pos_embed_dim]
        e_pos_b = self.embedding_pos_b(pos_idx)
        # flatten => [B, N, 2*64]
        B, N, D, C = e_pos_b.shape
        e_pos_flat = e_pos_b.view(B, N, D * C)
        # 线性层 => [B, N, d_model]
        e_pos = self.pos_linear(e_pos_flat)

        # =============== Part D: 三者相加 ===============
        # [B, N, 256]
        e = e_cmd + e_param + e_pos
        return e
