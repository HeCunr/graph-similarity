# model/GeomLayers/GeomAlignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAlignmentHead(nn.Module):
    """
    用于跨视图的节点对齐 + 节点级对比损失（与 Seq 项目中对比损失的写法一致）。
    假设输入均为 [B, N, d_model]。
    """
    def __init__(self, d_model: int, alignment='concat', perspectives=256, tau=0.07):
        super().__init__()
        self.d_model = d_model
        self.alignment = alignment
        self.perspectives = perspectives
        self.tau = tau

        # 用于对齐后的 MLP 投影 (可选)
        if alignment.lower() == 'concat':
            # concat后维度=2*d_model -> 再映射回 d_model
            self.alignment_layer = nn.Linear(2*d_model, d_model)
        elif alignment.lower() == 'bilinear':
            # TODO: 实现 bilinear
            raise NotImplementedError("bilinear alignment not implemented.")
        else:
            raise NotImplementedError(f"Unknown alignment={alignment}")

        # Contrastive投影头 (和原先一样, 用于做最终对比)
        hidden_dim = d_model * 2
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        # 与 Seq 项目中相同的多分类交叉熵损失
        self.ce = nn.CrossEntropyLoss()

    def perform_alignment(self, z_view1: torch.Tensor, z_view2: torch.Tensor):
        """
        基于余弦相似的注意力，对 v1,v2 做加权均值，然后与原向量 concat 产生对齐后的表示
        z_view1,z_view2: [B,N,d_model]
        return: out1, out2 => [B,N,d_model] (对齐后)
        """
        # 1) compute attention = cos( v1_i, v2_j )
        attention = self.node_alignment_attention(z_view1, z_view2)  # [B,N,N]

        # 2) 用 attention 加权 v2 => v1
        att_v2 = torch.bmm(attention, z_view2)  # [B,N,d_model]
        #  同理 v1 => v2
        att_v1 = torch.bmm(attention.transpose(1,2), z_view1)

        # 3) 按 alignment 模式融合
        if self.alignment.lower() == 'concat':
            merged1 = torch.cat([z_view1, att_v2], dim=-1)  # [B,N,2d]
            merged2 = torch.cat([z_view2, att_v1], dim=-1)
            out1 = self.alignment_layer(merged1)  # => [B,N,d_model]
            out2 = self.alignment_layer(merged2)
            return out1, out2
        else:
            raise NotImplementedError()

    def node_alignment_attention(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise node alignment attention [B,N,N], = cos(v1_i,v2_j)."""
        v1_norm = F.normalize(v1, dim=-1)   # [B,N,d]
        v2_norm = F.normalize(v2, dim=-1)   # [B,N,d]
        att = torch.bmm(v1_norm, v2_norm.transpose(1,2))  # [B,N,N]
        return att  # ∈ [-1,1]

    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        参考 Seq 项目的写法，将 (z1, z2) 先投影并拼接，然后构建正负样本对，最后用交叉熵。

        输入:
          z1, z2: [B, N, d_model]
        过程:
          1) proj_head -> (B,N,d_model)
          2) flatten -> (B*N, d_model)
          3) 拼成 2*(B*N) 大小的特征矩阵 => 计算相似度矩阵 => 构造 label_matrix
          4) 分离正负样本 => 拼 logits => 交叉熵 => 返回标量 loss
        """
        B, N, d = z1.shape
        device = z1.device

        # 1) 先投影
        p1 = self.proj_head(z1)  # [B,N,d]
        p2 = self.proj_head(z2)  # [B,N,d]

        # 2) flatten => [B*N, d]
        p1 = p1.view(B*N, d)
        p2 = p2.view(B*N, d)

        # 3) 规范化 (与 Seq 项目一致)
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        # 4) 拼合在一起 => [2*B*N, d]
        features = torch.cat([p1, p2], dim=0)  # => (2*B*N, d)
        total_size = features.size(0)         # 2*B*N

        # 5) 相似度矩阵 => shape [2*B*N, 2*B*N]
        similarity_matrix = torch.matmul(features, features.t())

        # 构造 label_matrix：前半部分 (p1) 的正样本对应后半部分 (p2) 的同索引
        # 即 i 与 i + B*N 对应
        label_matrix = torch.zeros(total_size, total_size, device=device, dtype=torch.bool)
        # 对角块设置为 True
        # p1 的第 i 行 对 p2 的第 i 行 => i 与 i+B*N
        half = B*N
        eye_mat = torch.eye(half, dtype=torch.bool, device=device)
        label_matrix[:half, half:] = eye_mat
        label_matrix[half:, :half] = eye_mat

        # 6) 去掉主对角线 (因为自己与自己对比无意义)
        mask = torch.eye(total_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix[~mask].view(total_size, -1)
        label_matrix = label_matrix[~mask].view(total_size, -1)

        # 7) positives / negatives 拆分
        positives = similarity_matrix[label_matrix]      # 所有正例
        negatives = similarity_matrix[~label_matrix]     # 所有负例

        # 重新 reshape，让每一行对应 "1 个正例 + 剩余负例"
        positives = positives.view(total_size, 1)        # 每行只有 1 个正例
        negatives = negatives.view(total_size, -1)       # 剩余负例
        logits = torch.cat([positives, negatives], dim=1)  # => [total_size, 1 + X]

        # 8) labels 全 0（表示第0列是正例）
        labels = torch.zeros(total_size, dtype=torch.long, device=device)

        # 9) 温度缩放 & 交叉熵
        logits = logits / self.tau
        cl_loss = self.ce(logits, labels)

        return cl_loss
