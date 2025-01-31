# model/GeomLayers/NodeAggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAggregator(nn.Module):
    """
    单次聚合子模块: [B,N,F] => [B,K,F], 同时更新邻接 [B,N,N] => [B,K,K].
    全程使用 float32.
    """
    def __init__(self, in_features: int, in_nodes: int, out_nodes: int):
        super().__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_features = in_features

        # 这里用一个简单的两层 MLP
        self.assign_mlp = nn.Sequential(
            nn.Linear(self.in_features, self.out_nodes),
            nn.ReLU(),
            nn.Linear(self.out_nodes, self.out_nodes)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        """
        x: [B, N, F], float32
        adj: [B, N, N], float32
        mask: [B, N],   float32
        return: pfeat [B, K, F], padj [B, K, K], pmask [B, K]
        """
        B, N, feat_dim = x.size()
        # 1) x reshape => [B*N, F]
        x_2d = x.reshape(B*N, feat_dim)  # float32
        logits_2d = self.assign_mlp(x_2d)  # => [B*N, K], float32

        # 2) 根据 mask => -1e9
        # mask 形状 [B,N]
        mask_1d = mask.view(B*N,1)  # float32
        large_neg = -1e9*(1 - mask_1d)
        logits_2d = logits_2d + large_neg

        # 3) softmax => [B*N,K]
        assign_2d = F.softmax(logits_2d, dim=1)

        # 4) reshape => [B,N,K]
        assign = assign_2d.view(B,N,self.out_nodes)

        # pooled_x = S^T x = [B,K,F]
        S_t = assign.transpose(1,2)  # => [B,K,N]
        pooled_x = torch.bmm(S_t, x)  # => [B,K,F]

        # pooled_adj = S^T A S => [B,K,K]
        mid = torch.bmm(adj, assign)    # [B,N,K]
        pooled_adj = torch.bmm(S_t, mid) # [B,K,K]

        # pooled_mask: 全1
        pmask = torch.ones((B, self.out_nodes), dtype=torch.float32, device=x.device)
        return pooled_x, pooled_adj, pmask


class MultiLevelNodeAggregator(nn.Module):
    """
    多层级聚合: 4096->2048->1024->512->256->128, float32.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.agg1 = NodeAggregator(in_features, 4096, 2048)
        self.agg2 = NodeAggregator(in_features, 2048, 1024)
        self.agg3 = NodeAggregator(in_features, 1024, 512)
        self.agg4 = NodeAggregator(in_features, 512, 256)
        self.agg5 = NodeAggregator(in_features, 256, 128)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        # x, adj, mask => float32
        x1, adj1, mask1 = self.agg1(x, adj, mask)
        x2, adj2, mask2 = self.agg2(x1, adj1, mask1)
        x3, adj3, mask3 = self.agg3(x2, adj2, mask2)
        x4, adj4, mask4 = self.agg4(x3, adj3, mask3)
        x5, adj5, mask5 = self.agg5(x4, adj4, mask4)
        return x5, adj5, mask5
