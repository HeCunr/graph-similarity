# model/GeomLayers/SparseMPNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv

class SparseMPNN(nn.Module):
    """
    简化版本的 GatedGraphConv, 针对 batch 里 B 张图做循环。
    out_channels = d_model
    """

    def __init__(self, out_channels: int, num_layers: int = 1):
        super().__init__()
        # 直接用PyG原生GatedGraphConv(num_layers=xxx)
        self.gnn = GatedGraphConv(
            out_channels=out_channels,
            num_layers=num_layers,
            aggr='add'
        )

    def forward(self, x, edge_index_list, edge_weight_list, mask):
        """
        x: [B, N, d_model]
        edge_index_list: list of length B, each: [2, E]
        edge_weight_list: list of length B, each: [E] or None
        mask: [B, N]
        return: updated_x [B, N, d_model]
        """
        B, N, d = x.shape
        out = torch.zeros_like(x)

        for i in range(B):
            valid_nodes = (mask[i] > 0).nonzero(as_tuple=True)[0]
            x_i = x[i, valid_nodes]
            ei = edge_index_list[i].to(x.device)
            ew = edge_weight_list[i].to(x.device) if edge_weight_list[i] is not None else None

            if x_i.size(0) == 0 or ei.numel() == 0:
                continue

            # [修改处] ================================
            # 对 edge_index 做 reindex
            idx_map = torch.full((N,), -1, device=x.device, dtype=torch.long)
            idx_map[valid_nodes] = torch.arange(valid_nodes.size(0), device=x.device)

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
            # ========================================

            updated_i = self.gnn(x_i, ei2, edge_weight=ew2)
            out[i, valid_nodes] = updated_i

        return out
