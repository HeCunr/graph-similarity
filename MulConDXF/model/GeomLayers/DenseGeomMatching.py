# model/GeomLayers/DenseGeomMatching.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
from torch_geometric.nn.dense.dense_gin_conv import DenseGINConv
from torch_geometric.nn.dense.dense_sage_conv import DenseSAGEConv
from model.GeomLayers.JKPooling import JKPooling

from model.GeomLayers.DenseGGNN import DenseGGNN
import numpy as np

class GraphMatchNetwork(nn.Module):
    """
    Graph Matching Network with contrastive learning (node-level).
    No final aggregator: we keep [B, N, perspectives] for InfoNCE.
    """
    def __init__(self, node_init_dims: int, args):
        super(GraphMatchNetwork, self).__init__()
        self.args = args
        self.node_init_dims = node_init_dims
        self.tau = args.tau
        self.dropout = args.dropout

        # Parse GNN filter configurations
        self.filters = [int(f) for f in args.filters.split('_')]
        self.num_layers = len(self.filters)
        self.last_filter = self.filters[-1]

        # Initialize GNN layers
        self.gnn_layers = self._build_gnn_layers()

        # 用于node-level InfoNCE的投影头
        self.proj_head = nn.Sequential(
            nn.Linear(self.last_filter, self.last_filter).double(),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_filter, self.last_filter).double()
        )

        # Matching layer (concat/bilinear)
        if args.match.lower() == 'concat':
            self.match_layer = ConcatMatchingLayer(self.last_filter, args.perspectives)
        elif args.match.lower() == 'bilinear':
            self.match_layer = BilinearMatchingLayer(self.last_filter, args.perspectives)
        else:
            raise NotImplementedError(f"Matching method {args.match} not implemented")

        # === 新增: JKPooling ===
        self.jk_pool = JKPooling(layer_dims=self.filters, mode='concat')  # 例: 100_100_100 => [100,100,100]

        # === 新增: 用于Geom->Seq对比的投影到对齐维度(可与node-level的proj_head大小不同)
        # 假设为了与Seq那边对齐，我们投影到 256 维，然后再做 L2 Norm
        self.geom_seq_proj = nn.Linear(sum(self.filters), 256).double()

    def _build_gnn_layers(self) -> nn.ModuleList:
        """Build GNN layers based on configuration"""
        layers = nn.ModuleList()

        # Prepare param sets for each GNN layer
        gcn_params = []
        for i in range(self.num_layers):
            in_channels = self.node_init_dims if i == 0 else self.filters[i - 1]
            out_channels = self.filters[i]
            gcn_params.append(dict(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=True
            ))

        gin_params = []
        for i in range(self.num_layers):
            in_channels = self.node_init_dims if i == 0 else self.filters[i - 1]
            out_channels = self.filters[i]
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels).double(),
                nn.ReLU()
            )
            gin_params.append(dict(nn=gin_mlp))

        layer_types = {
            'gcn': (DenseGCNConv, gcn_params),
            'graphsage': (DenseSAGEConv, gcn_params),
            'gin': (DenseGINConv, gin_params),
            'ggnn': (DenseGGNN, [dict(out_channels=f) for f in self.filters])
        }

        layer_class, params = layer_types[self.args.conv.lower()]

        for i in range(self.num_layers):
            layer = layer_class(**params[i])
            layer = layer.double()
            layers.append(layer)

        return layers

    def forward(self, x, adj, mask=None, collect_intermediate=False):
        # x: [B,N,D]
        all_layers = []
        for i, layer in enumerate(self.gnn_layers):
            if isinstance(layer, DenseGGNN):
                out = layer(x, adj, mask=mask)
            else:
                out = layer(x, adj, mask=mask, add_loop=False)

            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x = out
            if collect_intermediate:
                all_layers.append(x)

        if collect_intermediate:
            return x, all_layers
        else:
            return x

    def get_graph_repr_for_fusion(self, all_layer_outputs, mask=None):
        """
        使用 JKPooling 得到[B, sum_of_dims] => 线性映射到 256 => L2 normalize => [B,256]
        """
        # JKPooling
        graph_feat = self.jk_pool(all_layer_outputs, masks=mask)  # [B, sum_of_filters]

        # 再做线性映射
        graph_feat = self.geom_seq_proj(graph_feat)  # => [B,256]
        # 再做 L2
        graph_feat = F.normalize(graph_feat, dim=-1)
        return graph_feat

    def drop_feature(self, x: torch.Tensor, drop_prob: float) -> torch.Tensor:
        """Feature dropout for data augmentation"""
        x = x.double()
        for i in range(x.size(0)):
            drop_mask = torch.empty(
                (x[i].size(1),),
                dtype=torch.float64,
                device=x[i].device
            ).uniform_(0, 1) < drop_prob
            x[i] = x[i].clone()
            x[i][:, drop_mask] = 0
        return x

    def aug_random_edge(self, input_adj: np.ndarray, drop_percent: float) -> torch.Tensor:
        """Random edge augmentation"""
        drop_percent = drop_percent / 2
        b = np.where(
            input_adj > 0,
            np.random.choice(2, input_adj.shape, p=[drop_percent, 1 - drop_percent]),
            input_adj
        )
        drop_num = len(input_adj.nonzero()[0]) - len(b.nonzero()[0])
        mask_p = drop_num / (input_adj.shape[0] * input_adj.shape[0] - len(b.nonzero()[0]))
        c = np.where(b == 0,
                     np.random.choice(2, input_adj.shape, p=[1 - mask_p, mask_p]),
                     b)
        return torch.from_numpy(c).double()

    def matching_layer(self, feature_p: torch.Tensor, feature_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-graph node matching. Returns (matched_p, matched_h), each [B, N, perspectives].
        """
        feature_p = feature_p.double()
        feature_h = feature_h.double()

        # 计算 pairwise cosine
        attention = self.cosine_attention(feature_p, feature_h)

        # 加权
        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(-1)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(-1)

        # 加权均值
        att_mean_h = self.div_with_small_value(
            attention_h.sum(dim=2),
            attention.sum(dim=2, keepdim=True)
        )
        att_mean_p = self.div_with_small_value(
            attention_p.sum(dim=1),
            attention.sum(dim=1, keepdim=True).permute(0, 2, 1)
        )

        # Concat 并做匹配映射
        if self.args.match.lower() == 'concat':
            concat_p = torch.cat((feature_p, att_mean_h), dim=-1)  # [B, N, 2*D]
            concat_h = torch.cat((feature_h, att_mean_p), dim=-1)  # [B, N, 2*D]

            multi_p = self.match_layer(concat_p)  # [B, N, perspectives]
            multi_h = self.match_layer(concat_h)  # [B, N, perspectives]
        else:
            raise NotImplementedError(f"Matching method {self.args.match} not implemented")

        return multi_p, multi_h

    def cosine_attention(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine attention [B, N, N]."""
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    @staticmethod
    def div_with_small_value(n: torch.Tensor, d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int = 0, mean: bool = True) -> torch.Tensor:
        """
        InfoNCE-like contrastive loss on [B, N, perspectives].
        """
        ret = []
        length = z1.size(0)
        for i in range(length):
            # proj_head => [N, last_filter], N=节点数, last_filter=与 perspectives 相对应
            h1 = self.proj_head(z1[i])  # [N, last_filter]
            h2 = self.proj_head(z2[i])  # [N, last_filter]

            if batch_size == 0:
                l1 = self.semi_loss(h1, h2)
                l2 = self.semi_loss(h2, h1)
            else:
                l1 = self.batched_semi_loss(h1, h2, batch_size)
                l2 = self.batched_semi_loss(h2, h1, batch_size)

            ret_temp = (l1 + l2) * 0.5
            ret_temp = ret_temp.mean() if mean else ret_temp.sum()
            ret.append(ret_temp)

        return sum(ret) / len(ret)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.mm(z1, z2.t())  # [N, N]

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))       # [N, N]
        between_sim = f(self.sim(z1, z2))    # [N, N]

        return -torch.log(
            between_sim.diag() /
            (refl_sim.sum(dim=1) + between_sim.sum(dim=1) - refl_sim.diag())
        )

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        """可选的分段计算 if needed"""
        raise NotImplementedError()


class ConcatMatchingLayer(nn.Module):
    """Concatenation-based node matching layer"""
    def __init__(self, hidden_dim: int, num_perspectives: int):
        super(ConcatMatchingLayer, self).__init__()
        self.matcher = nn.Linear(2 * hidden_dim, num_perspectives).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 2*hidden_dim], output: [B, N, perspectives]
        return self.matcher(x)


class BilinearMatchingLayer(nn.Module):
    """Bilinear node matching layer"""
    def __init__(self, hidden_dim: int, num_perspectives: int):
        super(BilinearMatchingLayer, self).__init__()
        self.matcher = nn.Bilinear(hidden_dim, hidden_dim, num_perspectives).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果是 'bilinear'，需要事先分拆 x
        # 这里仅作示例，与 'concat' 方式不同
        raise NotImplementedError("Implement bilinear matching if needed.")
