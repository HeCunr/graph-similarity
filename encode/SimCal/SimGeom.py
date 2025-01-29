# SimGeom.py
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import os

# 如果项目结构允许，直接这样导入
# 如果你的项目是一个包结构，需要根据实际情况修改 import 路径
from model.GeomLayers.DenseGeomMatching import GraphMatchNetwork
from model.GeomLayers.NodeAggregator import MultiLevelNodeAggregator
from utils.Geom_utils import set_seed, get_device

###############################################################################
# 1) 解析命令行参数
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Compute similarity of two JSON graphs using trained Geom model (without matching layer).")
    parser.add_argument('--json1', type=str, default="/home/vllm/encode/data/Geom/TEST_4096/QFN28LK(Cu)-90-450 Rev1_5.json",
                        help="Path to the first JSON file (graph).")
    parser.add_argument('--json2', type=str, default="/home/vllm/encode/data/Geom/TEST_4096/QFN28LK(Cu)-90-450 Rev1_5.json",
                        help="Path to the second JSON file (graph).")
    parser.add_argument('--checkpoint', type=str, default="/home/vllm/encode/logs/Geom/PROTEINS_20250126_101652/checkpoints/batch_size16_epochs50.pt",
                        help="Path to the trained checkpoint file.")
    parser.add_argument('--graph_init_dim', type=int, default=44,
                        help="Node init dim (must match training).")
    parser.add_argument('--filters', type=str, default='100_100_100',
                        help="GNN hidden dimensions used in training.")
    parser.add_argument('--conv', type=str, default='ggnn',
                        help="Type of GNN layer used in training.")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate (must match training).")
    parser.add_argument('--tau', type=float, default=0.7,
                        help="Temperature param for InfoNCE (保持一致即可).")
    parser.add_argument('--gpu_index', type=str, default='0',
                        help="GPU index, set to '-1' or an invalid index to use CPU.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed.")

    # 补充的两个参数：match & perspectives，必须与训练时保持一致
    parser.add_argument('--match', type=str, default='concat',
                        help="Node matching method (concat/bilinear). Must match the training config.")
    parser.add_argument('--perspectives', type=int, default=100,
                        help="Number of perspectives for matching. Must match the training config.")

    args = parser.parse_args()
    return args

###############################################################################
# 2) 读取单个 JSON 文件并转成 (feature, adj, mask)
###############################################################################
def load_graph_from_json(json_path: str, max_n: int = 4096, feat_dim: int = 44):
    """
    假设 JSON 文件中含有:
    {
      "n_num": int,
      "src": str,    # 可选
      "features": 2D list (形状 [n_num, feat_dim]),
      "succs": list of lists, 表示每个节点的邻居
    }
    如果文件中有多行，则只取第一行做演示。

    返回:
      features, adj, mask 的 numpy 数组，形状分别是:
      [1, max_n, feat_dim], [1, max_n, max_n], [1, max_n]
    （外面加个 batch 维度 1）
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file {json_path} not found.")

    with open(json_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"JSON file {json_path} is empty.")
        g_info = json.loads(line)

    n_num = g_info["n_num"]
    if n_num <= 0:
        raise ValueError(f"Graph {json_path} has n_num={n_num} <= 0.")
    # 读取特征
    if "features" not in g_info:
        raise ValueError(f"Graph {json_path} missing 'features' field.")
    features = np.array(g_info["features"], dtype=np.float32)
    if features.ndim == 1:
        # 若是 1D，reshape
        features = features.reshape(1, -1)
    if features.shape[1] != feat_dim:
        raise ValueError(f"Graph {json_path} feature dim mismatch, expect {feat_dim}, got {features.shape[1]}")

    # 构建邻接: n_num x n_num
    adj = np.zeros((n_num, n_num), dtype=np.float32)
    succs = g_info["succs"]
    if len(succs) < n_num:
        raise ValueError(f"Graph {json_path} has succs len < n_num={n_num}")

    has_edge = False
    for u in range(n_num):
        for v in succs[u]:
            adj[u, v] = 1
            adj[v, u] = 1
            has_edge = True
    if not has_edge:
        raise ValueError(f"Graph {json_path} has no edges at all.")

    # 加 self-loop
    np.fill_diagonal(adj, 1)

    # 如果实际节点数 > max_n，需要截断
    if n_num > max_n:
        print(f"Warning: Graph {json_path} has {n_num} nodes, > max_n={max_n}, slicing to {max_n}")
        adj = adj[:max_n, :max_n]
        features = features[:max_n, :]
        n_num = max_n

    # 填充到 (max_n, max_n) / (max_n, feat_dim)，并生成 mask
    adj_padded = np.zeros((max_n, max_n), dtype=np.float32)
    adj_padded[:n_num, :n_num] = adj[:n_num, :n_num]

    feat_padded = np.full((max_n, feat_dim), -1, dtype=np.float32)
    feat_padded[:n_num, :] = features[:n_num, :]

    mask = np.zeros((max_n,), dtype=np.float32)
    mask[:n_num] = 1.

    # 最外层再加一个 batch 维: [1, max_n, ...]
    feat_padded = np.expand_dims(feat_padded, axis=0)  # [1, max_n, feat_dim]
    adj_padded = np.expand_dims(adj_padded, axis=0)    # [1, max_n, max_n]
    mask = np.expand_dims(mask, axis=0)               # [1, max_n]

    return feat_padded, adj_padded, mask

###############################################################################
# 3) 使用 pooling_module + model 得到图的向量表示（不走 matching layer）
###############################################################################
def get_graph_embedding(feat, adj, mask, pooling_module, model, device):
    """
    输入： 单个图的 (feat, adj, mask)，形状 [1, N, F], [1, N, N], [1, N]
    返回： 该图的最终 embedding 向量 [hidden_dim]（先均值再归一化）
    """
    # 转成 torch 张量
    feat_t = torch.from_numpy(feat).to(device, dtype=torch.float64)  # [1, N, F]
    adj_t = torch.from_numpy(adj).to(device, dtype=torch.float64)    # [1, N, N]
    mask_t = torch.from_numpy(mask).to(device, dtype=torch.float64)  # [1, N]

    # 1) 多层节点聚合
    with torch.no_grad():
        pfeat, padj, pmask = pooling_module(feat_t, adj_t, mask_t)   # [1, K, F], [1, K, K], [1, K]

    # 2) GNN 前向（不调用 matching_layer）
    with torch.no_grad():
        z = model(pfeat, padj, pmask)  # [1, K, last_filter]

    # 3) 这里简单做一个图级汇聚：对所有节点做均值，然后再做归一化
    z_avg = z.mean(dim=1)             # [1, last_filter]
    z_avg = F.normalize(z_avg, dim=-1)  # [1, last_filter], 单位向量

    return z_avg.squeeze(0)  # [last_filter]

###############################################################################
# 4) 主逻辑：加载模型、处理两个json、计算相似度并输出
###############################################################################
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args)

    # 初始化 pooling_module 和 model
    pooling_module = MultiLevelNodeAggregator(in_features=args.graph_init_dim).to(device)
    model = GraphMatchNetwork(node_init_dims=args.graph_init_dim, args=args).to(device)

    # 从 checkpoint 加载权重（假设其中有 'pooling_module_state_dict'）
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'pooling_module_state_dict' in checkpoint:
        pooling_module.load_state_dict(checkpoint['pooling_module_state_dict'])
    else:
        raise ValueError("Checkpoint does not contain pooling_module_state_dict. Please verify your saved file.")

    model.eval()
    pooling_module.eval()

    # 读取第一个图
    feat1, adj1, mask1 = load_graph_from_json(args.json1, max_n=4096, feat_dim=args.graph_init_dim)
    emb1 = get_graph_embedding(feat1, adj1, mask1, pooling_module, model, device)

    # 读取第二个图
    feat2, adj2, mask2 = load_graph_from_json(args.json2, max_n=4096, feat_dim=args.graph_init_dim)
    emb2 = get_graph_embedding(feat2, adj2, mask2, pooling_module, model, device)

    # 计算两图向量的余弦相似度（范围[-1,1]）
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=-1).item()

    # 将 [-1, 1] 映射到 [0, 1]
    normalized_sim = (similarity + 1.0) / 2.0

    print(f"Raw Cosine Similarity = {similarity:.4f}")
    print(f"Normalized Similarity = {normalized_sim:.4f} (range [0,1])")

if __name__ == "__main__":
    main()
