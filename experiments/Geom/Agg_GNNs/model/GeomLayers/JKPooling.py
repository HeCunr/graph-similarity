# model/GeomLayers/JKPooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class JKPooling(nn.Module):
    """
    A simple Jumping Knowledge style pooling:
    Collect node embeddings from each GNN layer, e.g. [B,N,d_l1], [B,N,d_l2], ...
    Then sum/concat -> [B,N,D], do a node-pooling -> [B,D].
    """
    def __init__(self, layer_dims, mode='concat'):
        super().__init__()
        self.layer_dims = layer_dims
        self.mode = mode

        if mode == 'concat':
            total_dim = sum(layer_dims)
        elif mode == 'sum':
            total_dim = layer_dims[-1]
        else:
            raise ValueError("Unsupported JK mode")

        self.fc = nn.Linear(total_dim, total_dim)  # float
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, all_layer_nodes, masks=None):
        """
        all_layer_nodes: list of Tensors, each [B,N,d_i] (float32)
        masks: [B,N], optional (float32)
        returns: [B, total_dim]
        """
        if self.mode == 'concat':
            cat_h = torch.cat(all_layer_nodes, dim=2)  # => [B,N, sum_of_dims]
        else:
            cat_h = 0
            for h in all_layer_nodes:
                cat_h = cat_h + h

        if masks is not None:
            cat_h = cat_h * masks.unsqueeze(-1)
            sum_h = cat_h.sum(dim=1)  # [B, sum_of_dims]
            valid_counts = masks.sum(dim=1, keepdim=True).clamp(min=1e-9)
            graph_h = sum_h / valid_counts
        else:
            graph_h = cat_h.mean(dim=1)

        out = self.fc(graph_h)  # => float
        out = self.act(out)
        out = self.dropout(out)
        return out
