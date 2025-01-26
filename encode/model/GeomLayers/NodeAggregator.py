# model/GeomLayers/NodeAggregator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAggregator(nn.Module):
    """
    单次聚合子模块: [B,N,F] => [B,K,F], 同时更新邻接 [B,N,N] => [B,K,K].
    这里用 double 以配合外部使用 double 的张量.
    """

    def __init__(self, in_features: int, in_nodes: int, out_nodes: int):
        super().__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_features = in_features

        # 这里用一个简单的两层 MLP, 并设置为 double
        self.assign_mlp = nn.Sequential(
            nn.Linear(self.in_features, self.out_nodes).double(),
            nn.ReLU(),
            nn.Linear(self.out_nodes, self.out_nodes).double()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        """
        x:   [B, N, F],   dtype = torch.float64
        adj: [B, N, N],   dtype = torch.float64
        mask:[B, N],      dtype = torch.float64 (或 bool)
        return: pfeat: [B, K, F], padj: [B, K, K], pmask: [B, K], 都为 double
        """
        B, N, feat_dim = x.size()
        device = x.device

        # 1) 先将 x reshape 成 [B*N, F], 送入 MLP => [B*N, K]
        # 这里 x, adj, mask 都是 double, MLP 也是 double
        x_2d = x.reshape(B*N, feat_dim)  # shape [B*N, F]
        logits_2d = self.assign_mlp(x_2d)  # [B*N, K], double

        # 2) 根据 mask，对无效节点加 -1e9
        # 注意 mask 可能是 float64, 需要先转 bool 或保持为 0/1
        # 若 mask 仅0/1，(1 - mask)也是0/1 => broadcast ok
        mask_1d = mask.view(B*N, 1)  # [B*N, 1], double
        large_neg = -1e9 * (1 - mask_1d)  # 仍是 double
        logits_2d = logits_2d + large_neg

        # 3) softmax => [B*N, K] (double)
        assign_2d = F.softmax(logits_2d, dim=1)

        # 4) reshape回 [B, N, K]
        assign = assign_2d.view(B, N, self.out_nodes)  # double

        # ========== 构建 pooled_x = S^T x = (B, K, F) ==========

        S_t = assign.transpose(1, 2)  # [B, K, N], double
        pooled_x = torch.bmm(S_t, x)  # [B, K, F], double

        # ========== 构建 pooled_adj = S^T A S = (B, K, K) ==========
        mid = torch.bmm(adj, assign)     # [B, N, K], double
        pooled_adj = torch.bmm(S_t, mid) # [B, K, K], double

        # 3) pooled_mask: 全 1 => dtype 与 x 保持一致 (double)
        pmask = torch.ones((B, self.out_nodes), dtype=torch.float64, device=device)

        return pooled_x, pooled_adj, pmask


class MultiLevelNodeAggregator(nn.Module):
    """
    多层级聚合: 4096->2048->1024->512->256->128
    全程用 double.
    """

    def __init__(self, in_features: int):
        super().__init__()
        # 5 层 aggregator
        self.agg1 = NodeAggregator(in_features, 4096, 2048)
        self.agg2 = NodeAggregator(in_features, 2048, 1024)
        self.agg3 = NodeAggregator(in_features, 1024, 512)
        self.agg4 = NodeAggregator(in_features, 512, 256)
        self.agg5 = NodeAggregator(in_features, 256, 128)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        # 第1层: 4096->2048
        x1, adj1, mask1 = self.agg1(x, adj, mask)
        # 第2层: 2048->1024
        x2, adj2, mask2 = self.agg2(x1, adj1, mask1)
        # 第3层: 1024->512
        x3, adj3, mask3 = self.agg3(x2, adj2, mask2)
        # 第4层: 512->256
        x4, adj4, mask4 = self.agg4(x3, adj3, mask3)
        # 第5层: 256->128
        x5, adj5, mask5 = self.agg5(x4, adj4, mask4)

        return x5, adj5, mask5
