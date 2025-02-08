# model/GeomLayers/GeomGGNNBlock.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv, DenseGINConv
from model.GeomLayers.DenseGGNN import DenseGGNN

class GeomGGNNBlock(nn.Module):
    """
    根据 args.filters 指定的若干层，将图卷积 / GGNN 串起来构成一个Block。
    """
    def __init__(self, node_init_dims: int, args):
        super(GeomGGNNBlock, self).__init__()
        self.args = args
        self.dropout = args.dropout

        # 解析 filters，比如 "256_256_256" => [256, 256, 256]
        self.filters = [int(f) for f in args.filters.split('_')]
        self.num_layers = len(self.filters)
        # 记录最后一层输出维度（如有需要）
        self.last_filter = self.filters[-1]

        # 构建多层 GNN
        self.gnn_layers = self._build_gnn_layers()

    def _build_gnn_layers(self):
        layers = nn.ModuleList()
        conv_type = self.args.conv.lower()  # "gcn" / "graphsage" / "gin" / "ggnn"

        # 对于 gcn/graphsage/gin，我们需要 (in_channels, out_channels) 参数
        gcn_params = []
        gin_params = []
        for i in range(self.num_layers):
            in_channels = self.filters[i - 1] if i > 0 else self.args.graph_init_dim
            out_channels = self.filters[i]
            gcn_params.append(dict(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=True
            ))
            # gin需要一个 mlp
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
            gin_params.append(dict(nn=gin_mlp))

        if conv_type == 'gcn':
            for i in range(self.num_layers):
                layer = DenseGCNConv(**gcn_params[i])
                layers.append(layer)

        elif conv_type == 'graphsage':
            for i in range(self.num_layers):
                layer = DenseSAGEConv(**gcn_params[i])
                layers.append(layer)

        elif conv_type == 'gin':
            for i in range(self.num_layers):
                layer = DenseGINConv(**gin_params[i])
                layers.append(layer)

        elif conv_type == 'ggnn':
            # GGNN 的写法与前面略有区别
            # 根据原始示例，对于每层都构建一个 DenseGGNN(out_channels=filters[i])。
            # 也可以只构建一次，并把 num_layers=某值，但为和示例保持一致，这里一层一构建。
            for i in range(self.num_layers):
                out_channels = self.filters[i]
                # 每一层用 num_layers=1
                layer = DenseGGNN(out_channels=out_channels, num_layers=1)
                layers.append(layer)

        else:
            raise ValueError(f"Unsupported conv type: {self.args.conv}")

        return layers

    def forward(self, x, adj, mask=None, collect_intermediate=False):
        """
        x: [B, N, in_channels], 初次进入时 in_channels = args.graph_init_dim
        adj: [B, N, N]
        mask: [B, N] or None
        collect_intermediate: 是否收集每层输出（可选）
        """
        all_layers_out = []
        for i, layer in enumerate(self.gnn_layers):
            if isinstance(layer, DenseGGNN):
                out = layer(x, adj, mask=mask)
            else:
                # 注意: add_loop=False => 已包含自环
                out = layer(x, adj, mask=mask, add_loop=False)

            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x = out

            if collect_intermediate:
                all_layers_out.append(x)

        # 返回最后输出或中间层
        if collect_intermediate:
            return x, all_layers_out
        else:
            return x
