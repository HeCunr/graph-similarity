# model/GeomLayers/SparseNodeAggregator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class SparseNodeAggregator(nn.Module):
    def __init__(self, in_features: int, in_nodes: int, out_nodes: int):
        super().__init__()
        self.in_features = in_features
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        # 仅保留main分支
        self.assign_main = nn.Sequential(
            nn.Linear(in_features, out_nodes),
            nn.ReLU(),
            nn.Linear(out_nodes, out_nodes)
        )

    def forward(self, x, edge_index_list, edge_weight_list, mask):
        """
        x: [B, N, in_features]
        edge_index_list: len=B, each shape [2, E]
        edge_weight_list: len=B, each shape [E]
        mask: [B, N]

        Returns:
          pfeat: [B, out_nodes, in_features]
          p_edge_index_list: new list of [2, E']
          p_edge_weight_list: new list of [E']
          pmask: [B, out_nodes]
        """
        B, N, C = x.size()  # 将原本的 F 改成 C 或别的名称
        device = x.device

        out_feats = []
        out_eidxs = []
        out_ewgts = []
        out_masks = []

        for i in range(B):
            valid_nodes = (mask[i] > 0).nonzero(as_tuple=True)[0]
            x_i = x[i, valid_nodes]  # shape [n_i, C]
            n_i = x_i.size(0)
            if n_i == 0:
                # 空图 => 直接生成 out_nodes=0 ?
                pfeat_i = torch.zeros((self.out_nodes, C), device=device)
                pmask_i = torch.zeros((self.out_nodes,), device=device)
                p_ei = torch.empty((2, 0), dtype=torch.long, device=device)
                p_ew = torch.empty((0,), dtype=torch.float, device=device)

                out_feats.append(pfeat_i)
                out_masks.append(pmask_i)
                out_eidxs.append(p_ei)
                out_ewgts.append(p_ew)
                continue

            # 重映射
            ei = edge_index_list[i].to(device)
            ew = edge_weight_list[i].to(device) if edge_weight_list[i] is not None else None
            idx_map = torch.full((N,), -1, device=device, dtype=torch.long)
            idx_map[valid_nodes] = torch.arange(n_i, device=device)

            u, v = ei
            u2 = idx_map[u]
            v2 = idx_map[v]
            valid_mask = (u2 >= 0) & (v2 >= 0)
            u2 = u2[valid_mask]
            v2 = v2[valid_mask]
            ei2 = torch.stack([u2, v2], dim=0)

            if ew is not None:
                ew2 = ew[valid_mask]
            else:
                ew2 = None

            # ----(1) compute S_main----
            logits = self.assign_main(x_i)  # [n_i, out_nodes]
            S_main = F.softmax(logits, dim=1)  # [n_i, out_nodes]  <-- 现在 F 指向函数库，不会报错

            # ----(2) 聚合特征 pfeat = S^T * x_i----
            pfeat_i = torch.matmul(S_main.t(), x_i)  # [out_nodes, C]

            # ----(3) 构造新的邻接: pooled_adj = S^T * (A * S)----
            mid = torch.zeros((n_i, self.out_nodes), device=device)
            if ei2.size(1) > 0:
                _u, _v = ei2
                w = ew2 if ew2 is not None else torch.ones(ei2.size(1), device=device)
                w = w.unsqueeze(-1)  # shape [E, 1]
                S_u = S_main[_u]     # shape [E, out_nodes]
                src = w * S_u        # shape [E, out_nodes]
                scatter_add(src, _v, dim=0, out=mid)  # mid[v2] += ...

            pooled_adj = torch.matmul(S_main.t(), mid)  # [out_nodes, out_nodes]

            mask_nz = (pooled_adj.abs() > 1e-9)
            row, col = mask_nz.nonzero(as_tuple=True)
            val = pooled_adj[row, col]
            p_ei = torch.stack([row, col], dim=0)  # [2, E']
            p_ew = val

            # ----(4) pmask
            pmask_i = torch.ones((self.out_nodes,), device=device)

            out_feats.append(pfeat_i)
            out_masks.append(pmask_i)
            out_eidxs.append(p_ei)
            out_ewgts.append(p_ew)

        pfeat_list = []
        pmask_list = []
        for b in range(B):
            pfeat_b = out_feats[b]
            if pfeat_b.size(0) < self.out_nodes:
                pad_n = self.out_nodes - pfeat_b.size(0)
                pad_feat = torch.zeros((pad_n, C), device=device)
                pfeat_b = torch.cat([pfeat_b, pad_feat], dim=0)
            pfeat_list.append(pfeat_b.unsqueeze(0))

            pmask_b = out_masks[b]
            if pmask_b.size(0) < self.out_nodes:
                pad_m = torch.zeros((self.out_nodes - pmask_b.size(0),), device=device)
                pmask_b = torch.cat([pmask_b, pad_m], dim=0)
            pmask_list.append(pmask_b.unsqueeze(0))

        pfeat_out = torch.cat(pfeat_list, dim=0)  # [B, out_nodes, C]
        pmask_out = torch.cat(pmask_list, dim=0)  # [B, out_nodes]

        return pfeat_out, out_eidxs, out_ewgts, pmask_out
