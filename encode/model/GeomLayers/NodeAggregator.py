# model/GeomLayers/NodeAggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAggregator(nn.Module):
    def __init__(self, in_features: int, in_nodes: int, out_nodes: int):
        super().__init__()
        self.in_features = in_features
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        self.assign_main = nn.Sequential(
            nn.Linear(in_features, out_nodes),
            nn.ReLU(),
            nn.Linear(out_nodes, out_nodes)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        """
        x:   [B, N, F]
        adj: [B, N, N]
        mask:[B, N]
        """
        B, N, feat_dim = x.size()
        device = x.device

        # 1) 计算分配矩阵 (assignment matrix)
        #    先把 x reshape 成 [B*N, F]，再线性映射得到 [B*N, out_nodes]。
        x2d = x.view(B*N, feat_dim)
        logits_main = self.assign_main(x2d)  # => [B*N, out_nodes]

        # 根据 mask 将无效节点的 logits 设置为 -1e9，以保证 softmax 后归为 0
        mask_1d = mask.view(B*N, 1)
        large_neg = -1e9 * (1 - mask_1d)
        logits_main = logits_main + large_neg

        # softmax 获得归一化分配权重
        assign_main_2d = F.softmax(logits_main, dim=1)  # => [B*N, out_nodes]
        S_main = assign_main_2d.view(B, N, self.out_nodes)  # => [B, N, out_nodes]

        # 2) 用分配矩阵对特征和邻接做聚合
        #    pfeat_main = S^T * x
        pfeat_main = torch.bmm(S_main.transpose(1, 2), x)  # => [B, out_nodes, F]

        #    pooled_adj = S^T * adj * S
        mid = torch.bmm(adj, S_main)  # [B, N, out_nodes]
        pooled_adj = torch.bmm(S_main.transpose(1,2), mid)  # => [B, out_nodes, out_nodes]

        # 3) 新的 mask（聚合后的节点均有效）
        pmask = torch.ones((B, self.out_nodes), device=device)

        return pfeat_main, pooled_adj, pmask
