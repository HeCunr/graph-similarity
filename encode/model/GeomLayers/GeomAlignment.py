# model/GeomLayers/GeomAlignment.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAlignmentHead(nn.Module):
    """
    用于跨视图的节点对齐 + 节点级 InfoNCE。
    假设输入均为 [B, N, d_model]。
    """
    def __init__(self, d_model: int, alignment='concat', perspectives=128, tau=1.0):
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

        # Contrastive投影头 (用于 InfoNCE)
        hidden_dim = d_model * 2
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

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
        节点级 InfoNCE:
        1) 先投影 => [B,N,d_model]
        2) 逐batch、逐节点计算 InfoNCE
        """
        B, N, d = z1.shape
        # 投影
        p1 = self.proj_head(z1)  # [B,N,d]
        p2 = self.proj_head(z2)  # [B,N,d]

        # 合并 batch & node => [B*N,d]
        p1 = p1.reshape(B*N, d)
        p2 = p2.reshape(B*N, d)

        # 计算相似矩阵 [B*N, B*N]
        sim_matrix = self._sim(p1, p2)
        # 对角线 (i,i) => 正样本
        sim_diag = torch.diag(sim_matrix)  # [B*N]
        exp_diag = torch.exp(sim_diag / self.tau)
        sum_over_j = torch.sum(torch.exp(sim_matrix / self.tau), dim=1)  # [B*N]

        loss_i = -torch.log(exp_diag / sum_over_j)
        return loss_i.mean()

    def _sim(self, x: torch.Tensor, y: torch.Tensor):
        """Compute [batch_size, batch_size] similarity matrix"""
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.mm(x, y.transpose(0,1))  # [bs,bs]
