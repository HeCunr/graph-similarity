# model/layers/DenseGraphMatching.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import torch.nn.functional as F  # 添加此行

# 如果要对 'gcn'/'graphsage'/'gin' 也改成稀疏版本，需要改成:
# from torch_geometric.nn import GCNConv, SAGEConv, GINConv
# 并且完全重写 forward 中对 adj -> edge_index 的处理
# 这里先保留你原有的 DenseXXXConv
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
from torch_geometric.nn.dense.dense_gin_conv import DenseGINConv
from torch_geometric.nn.dense.dense_sage_conv import DenseSAGEConv

from model.layers.DenseGGNN import DenseGGNN

import numpy as np

class GraphMatchNetwork(nn.Module):
    """
    Graph Matching Network with contrastive learning for graph similarity computation
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

        # Initialize GNN layers (见下)
        self.gnn_layers = self._build_gnn_layers()

        # Initialize projection layers for contrastive learning with double precision
        self.proj_head = nn.Sequential(
            nn.Linear(self.last_filter, self.last_filter).double(),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_filter, self.last_filter).double()
        )

        # Initialize attention mechanism
        self.attention = CrossGraphAttention(self.last_filter)

        # Initialize matching layer
        if args.match.lower() == 'concat':
            self.match_layer = ConcatMatchingLayer(self.last_filter, args.perspectives)
        elif args.match.lower() == 'bilinear':
            self.match_layer = BilinearMatchingLayer(self.last_filter, args.perspectives)
        else:
            raise NotImplementedError(f"Matching method {args.match} not implemented")

        # Initialize aggregation
        if args.match_agg.lower() == 'lstm':
            self.aggregator = nn.LSTM(
                input_size=args.perspectives,
                hidden_size=args.hidden_size,
                batch_first=True
            )
        else:
            self.aggregator = None

    def _build_gnn_layers(self) -> nn.ModuleList:
        """Build GNN layers based on configuration"""
        layers = nn.ModuleList()

        # 根据层数解析 in/out channels
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
            # DenseGINConv 需要传一个 MLP (nn=...)
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels).double(),
                nn.ReLU()
            )
            gin_params.append(dict(nn=gin_mlp))

        # 如果是 ggnn，用我们修好的 DenseGGNN
        # 否则仍然使用 PyG 中的 DenseGCNConv / DenseSAGEConv / DenseGINConv
        layer_types = {
            'gcn': (DenseGCNConv, gcn_params),
            'graphsage': (DenseSAGEConv, gcn_params),
            'gin': (DenseGINConv, gin_params),
            'ggnn': (DenseGGNN, [dict(out_channels=f) for f in self.filters])
        }

        layer_class, params = layer_types[self.args.conv.lower()]

        # Build layers
        for i in range(self.num_layers):
            layer = layer_class(**params[i])
            layer = layer.double()
            layers.append(layer)

        return layers

    def gnn_forward(self, x, adj, mask=None):
        for layer in self.gnn_layers:
            x = layer(x, adj, add_loop=False)
            if mask is not None:
                x = x * mask.unsqueeze(-1)  # 屏蔽无效节点
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def drop_features(
            self,
            x: torch.Tensor,
            drop_prob: float
    ) -> torch.Tensor:
        """Apply feature dropout for data augmentation"""
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device
        ).uniform_(0, 1) < drop_prob

        x = x.clone()
        x[:, drop_mask] = 0
        return x

    def drop_edges(
            self,
            adj: torch.Tensor,
            drop_prob: float
    ) -> torch.Tensor:
        """Apply edge dropout for data augmentation"""
        adj = adj.clone()
        mask = torch.rand(adj.size(), device=adj.device) > drop_prob
        adj = adj * mask.float()
        return adj

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        如果是 'ggnn'，会走 DenseGGNN，自带 adj->edge_index 转换；
        如果是 'gcn'/'graphsage'/'gin'，则使用 DenseGCNConv... 需要依赖 dense adjacency。
        全零矩阵时，这些 dense layers 不会报错，只是做一次(0)加和 + 线性变换。
        """
        x = x.double()
        adj = adj.double()

        # 走 GNN forward
        for layer in self.gnn_layers:
            if isinstance(layer, DenseGGNN):
                # DenseGGNN 内部自动转换稠密邻接为稀疏
                out = layer(x, adj, mask=mask)
            else:
                # 原有 DenseGCNConv / DenseSAGEConv / DenseGINConv 走 dense 邻接
                # 这里演示可以手动加 self-loop 或不加，看 add_loop 参数
                # Dense 版通常自带 add_loop
                out = layer(x, adj, mask=mask, add_loop=False)

            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x = out

        return x.float()


    def drop_feature(self, x: torch.Tensor, drop_prob: float) -> torch.Tensor:
        """
        Drop node features for augmentation
        """
        x = x.double()  # Convert to double
        for i in range(x.size()[0]):
            drop_mask = torch.empty(
                (x[i].size(1),),
                dtype=torch.float64,  # Use float64 for double precision
                device=x[i].device
            ).uniform_(0, 1) < drop_prob
            x[i] = x[i].clone()
            x[i][:, drop_mask] = 0
        return x.float()  # Convert back to float

    def aug_random_edge(self, input_adj: np.ndarray, drop_percent: float) -> torch.Tensor:
        """
        Randomly augment edges
        """
        drop_percent = drop_percent / 2
        # Remove edges
        b = np.where(input_adj > 0,
                     np.random.choice(2, input_adj.shape, p=[drop_percent, 1 - drop_percent]),
                     input_adj)

        # Add new edges
        drop_num = len(input_adj.nonzero()[0]) - len(b.nonzero()[0])
        mask_p = drop_num / (input_adj.shape[0] * input_adj.shape[0] - len(b.nonzero()[0]))
        c = np.where(b == 0,
                     np.random.choice(2, input_adj.shape, p=[1 - mask_p, mask_p]),
                     b)

        # Convert to double precision tensor
        return torch.from_numpy(c).double()

    def matching_layer(self, feature_p: torch.Tensor, feature_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-graph matching layer

        Args:
            feature_p, feature_h: Node features [B, N, D]

        Returns:
            Tuple of matched features [B, N, P]
        """
        # Convert to double precision
        feature_p = feature_p.double()
        feature_h = feature_h.double()

        # Compute attention scores [B, N, N]
        attention = self.cosine_attention(feature_p, feature_h)

        # Weight node features by attention
        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(-1)  # [B, N, N, D]
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(-1)  # [B, N, N, D]

        # Compute weighted means [B, N, D]
        att_mean_h = self.div_with_small_value(
            attention_h.sum(dim=2),
            attention.sum(dim=2, keepdim=True)
        )
        att_mean_p = self.div_with_small_value(
            attention_p.sum(dim=1),
            attention.sum(dim=1, keepdim=True).permute(0, 2, 1)
        )

        # Apply matching layer
        if self.args.match.lower() == 'concat':
            # Concatenate features [B, N, 2D]
            concat_p = torch.cat((feature_p, att_mean_h), dim=-1)
            concat_h = torch.cat((feature_h, att_mean_p), dim=-1)

            # Get node-level matching scores [B, N, P]
            multi_p = self.match_layer(concat_p)
            multi_h = self.match_layer(concat_h)
        else:
            raise NotImplementedError(f"Matching method {self.args.match} not implemented")

        # Convert back to float
        return multi_p.float(), multi_h.float()

    def cosine_attention(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine attention scores
        """
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    @staticmethod
    def div_with_small_value(n: torch.Tensor, d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Safe division with small values
        """
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int = 0, mean: bool = True) -> torch.Tensor:
        """
        Compute contrastive loss between two views
        """
        ret = []
        length = z1.size()[0]
        for i in range(length):
            # Make sure inputs are double precision
            h1 = self.proj_head(z1[i].double())
            h2 = self.proj_head(z2[i].double())

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
        z1 = F.normalize(z1)  # 将 functional 替换为 F
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def match_and_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Match nodes and aggregate to graph-level representation"""
        # Apply matching layer
        matched = self.match_layer(x)

        # Aggregate based on configuration
        if self.aggregator is not None:
            output, _ = self.aggregator(matched)
            return output[:, -1]  # Take last LSTM output
        else:
            return torch.mean(matched, dim=1)  # Simple mean pooling

    def compute_loss(
            self,
            z1: Tuple[torch.Tensor, torch.Tensor],
            z2: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute contrastive loss between graph views

        Args:
            z1: Tuple of two views of first graph
            z2: Tuple of two views of second graph

        Returns:
            torch.Tensor: Contrastive loss value
        """
        z1_view1, z1_view2 = z1
        z2_view1, z2_view2 = z2

        # Compute similarities
        sim11 = self.similarity(z1_view1, z1_view2)
        sim22 = self.similarity(z2_view1, z2_view2)

        # Compute loss for each graph
        loss1 = self.contrastive_loss(sim11)
        loss2 = self.contrastive_loss(sim22)

        return (loss1 + loss2) / 2

    def similarity(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between embeddings"""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def contrastive_loss(self, sim: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss"""
        sim = sim / self.tau
        exp_sim = torch.exp(sim)
        pos_sim = torch.diag(exp_sim)
        loss = -torch.log(pos_sim / exp_sim.sum(dim=1))
        return loss.mean()


class CrossGraphAttention(nn.Module):
    """Cross-graph attention mechanism"""
    def __init__(self, hidden_dim: int):
        super(CrossGraphAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, hidden_dim)

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-graph attention

        Args:
            x1, x2: Node features from two graphs

        Returns:
            Updated node features for both graphs
        """
        # Compute attention scores
        scores = torch.bmm(
            self.attention(x1),
            x2.transpose(1, 2)
        )
        scores = F.softmax(scores, dim=-1)

        # Update node features
        x1_updated = torch.bmm(scores, x2)
        x2_updated = torch.bmm(scores.transpose(1, 2), x1)

        return x1_updated, x2_updated


class ConcatMatchingLayer(nn.Module):
    """Concatenation-based node matching layer"""
    def __init__(self, hidden_dim: int, num_perspectives: int):
        super(ConcatMatchingLayer, self).__init__()
        # Initialize matcher with double precision
        self.matcher = nn.Linear(2 * hidden_dim, num_perspectives).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of matching layer
        Args:
            x: input tensor of shape [B, N, D] with double precision
        Returns:
            output: tensor of shape [B, N, P]
        """
        # No need to convert to double since input should already be double
        return self.matcher(x)

class BilinearMatchingLayer(nn.Module):
    """Bilinear node matching layer"""
    def __init__(self, hidden_dim: int, num_perspectives: int):
        super(BilinearMatchingLayer, self).__init__()
        self.matcher = nn.Bilinear(hidden_dim, hidden_dim, num_perspectives).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: input tensor of shape [B, N, D] with double precision
        Returns:
            output: tensor of shape [B, N, P]
        """
        return self.matcher(x)