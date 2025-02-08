#model/GeomLayers/NodeAggregatorRes.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAggregatorRes(nn.Module):
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
        self.assign_res = nn.Sequential(
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
        # 不要写 B, N, F = x.size()，避免 F 覆盖 torch.nn.functional
        B, N, feat_dim = x.size()
        device = x.device

        # 1) main聚合
        x2d = x.view(B*N, feat_dim)
        logits_main = self.assign_main(x2d)  # => [B*N, K]

        mask_1d = mask.view(B*N, 1)
        large_neg = -1e9 * (1 - mask_1d)
        # logits_main是浮点tensor，这里 += large_neg 仍是浮点
        logits_main = logits_main + large_neg

        # 这里的F是 torch.nn.functional
        assign_main_2d = F.softmax(logits_main, dim=1)  # => [B*N, K]
        S_main = assign_main_2d.view(B, N, self.out_nodes)

        pfeat_main = torch.bmm(S_main.transpose(1,2), x)

        # 2) residual聚合
        logits_res = self.assign_res(x2d)
        logits_res = logits_res + large_neg
        assign_res_2d = F.softmax(logits_res, dim=1)
        S_res = assign_res_2d.view(B, N, self.out_nodes)
        pfeat_res = torch.bmm(S_res.transpose(1,2), x)

        pfeat = pfeat_main + pfeat_res

        # 3) 更新邻接
        mid = torch.bmm(adj, S_main)
        pooled_adj = torch.bmm(S_main.transpose(1,2), mid)

        # 4) 新的mask => 全1
        pmask = torch.ones((B, self.out_nodes), device=device)

        return pfeat, pooled_adj, pmask
