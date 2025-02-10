# model/GeomLayers/SparseGGNNBlock.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GatedGraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class SparseGGNNBlock(nn.Module):
    """
    类似原GeomGGNNBlock，但只针对稀疏edge_index。
    根据args.filters, args.conv构建多层。
    """

    def __init__(self, node_init_dims, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout

        self.filters = [int(f) for f in args.filters.split('_')]
        self.num_layers = len(self.filters)
        self.last_filter = self.filters[-1]

        # 构建多层
        self.gnn_layers = nn.ModuleList()
        self._build_layers(node_init_dims)

    def _build_layers(self, in_dim):
        conv_type = self.args.conv.lower()  # gcn / graphsage / gin / ggnn
        for i, out_dim in enumerate(self.filters):
            if i > 0:
                in_dim = self.filters[i-1]
            if conv_type == 'gcn':
                conv = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=False)
            elif conv_type == 'graphsage':
                conv = SAGEConv(in_dim, out_dim)
            elif conv_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                conv = GINConv(mlp)
            elif conv_type == 'ggnn':
                conv = GatedGraphConv(
                    out_dim, num_layers=1, aggr='add'
                )
            else:
                raise ValueError(f"Unsupported conv={conv_type}")
            self.gnn_layers.append(conv)

    def forward(self, x, edge_index_list, edge_weight_list, mask):
        """
        x: [B, N, in_dim]
        edge_index_list: list of length B, each [2, E]
        edge_weight_list:list of length B, each [E] or None
        mask: [B,N]
        return: x_out [B, N, last_filter]
        """
        B, N, _ = x.shape
        out = x
        for conv_layer in self.gnn_layers:
            out = self._forward_one_layer(conv_layer, out, edge_index_list, edge_weight_list, mask)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def _forward_one_layer(self, conv, x, edge_index_list, edge_weight_list, mask):
        """单层前向: batch里的每个图单独跑conv"""
        B, N, d = x.shape
        out = torch.zeros_like(x)

        for i in range(B):
            valid_nodes = (mask[i] > 0).nonzero(as_tuple=True)[0]
            x_i = x[i, valid_nodes]
            ei = edge_index_list[i]
            ew = edge_weight_list[i] if edge_weight_list[i] is not None else None

            if x_i.size(0) == 0 or ei.numel() == 0:
                continue

            # 如果是GCNConv，还需要在外部做norm
            # ei_n, ew_n = gcn_norm(ei, ew, x_i.size(0), add_self_loops=False)

            # [修改处] ================================
            # 重映射索引，让 ei 的端点与 x_i.size(0) 对应
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

            if isinstance(conv, GatedGraphConv):
                out_i = conv(x_i, ei2, edge_weight=ew2)
            else:
                out_i = conv(x_i, ei2, ew2)

            out[i, valid_nodes] = out_i
        return out
