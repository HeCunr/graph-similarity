# utils/Geom_utils.py

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from dataset.Geom_dataset import GraphData


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    简单的特征dropout：对每个样本，将若干维特征置0。
    x: [B, N, F]
    drop_prob: float
    """
    B, N, F = x.size()
    for i in range(B):
        drop_mask = torch.empty((F,), device=x.device).uniform_(0,1) < drop_prob
        x[i, :, drop_mask] = 0
    return x

def aug_random_edge(adj: torch.Tensor, drop_percent: float) -> torch.Tensor:
    """
    随机删除/添加部分边
    adj: [B, N, N]
    drop_percent: float
    """
    B, N, _ = adj.shape
    new_adj_list = []
    for i in range(B):
        arr = adj[i].cpu().numpy()
        drop_p = drop_percent/2
        # 1) 对现有边随机drop
        b = np.where(arr>0,
                     np.random.choice(2, arr.shape, p=[drop_p,1-drop_p]),
                     arr)
        # 2) 统计要再随机加多少
        drop_num = len(arr.nonzero()[0]) - len(b.nonzero()[0])
        total_potential = arr.size - len(b.nonzero()[0])
        mask_p = drop_num/total_potential if total_potential>0 else 0
        c = np.where(b==0,
                     np.random.choice(2, arr.shape, p=[1-mask_p, mask_p]),
                     b)
        new_adj_list.append(torch.from_numpy(c).float())
    new_adj = torch.stack(new_adj_list, dim=0).to(adj.device)
    return new_adj

def collate_graphs(batch: List[GraphData]):
    """
    输出:
    features: [B, max_n, feat_dim]
    adjs:     [B, max_n, max_n]
    masks:    [B, max_n]
    pos2ds:   [B, max_n, 2]
    同时返回一个 graph_names 用于调试打印等用途。
    """
    features = []
    adjs = []
    masks = []
    pos2ds = []
    graph_names = []

    for g in batch:
        f, a, m, p = g.matrices
        # 转为tensor
        features.append(torch.tensor(f, dtype=torch.float32))
        adjs.append(torch.tensor(a, dtype=torch.float32))
        masks.append(torch.tensor(m, dtype=torch.float32))
        pos2ds.append(torch.tensor(p, dtype=torch.float32))
        graph_names.append(g.name)

    # 堆叠
    features = torch.stack(features, dim=0)  # [B, max_n, feat_dim]
    adjs = torch.stack(adjs, dim=0)         # [B, max_n, max_n]
    masks = torch.stack(masks, dim=0)       # [B, max_n]
    pos2ds = torch.stack(pos2ds, dim=0)     # [B, max_n, 2]

    return features, adjs, masks, pos2ds, graph_names

def get_device(args) -> torch.device:
    if torch.cuda.is_available() and int(args.gpu_index) >= 0:
        device = torch.device(f'cuda:{args.gpu_index}')
        torch.cuda.set_device(int(args.gpu_index))
    else:
        device = torch.device('cpu')
    return device

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_single_graph_data(graph, device):
    if graph.matrices is None:
        raise ValueError("Graph has not been preprocessed")

    feat, adj_mat, mask_mat, pos2d_mat = graph.matrices  # 4项

    features = torch.tensor(feat, dtype=torch.float32, device=device)
    adj = torch.tensor(adj_mat, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_mat, dtype=torch.float32, device=device)
    pos2d = torch.tensor(pos2d_mat, dtype=torch.float32, device=device)

    return features, adj, mask, pos2d
