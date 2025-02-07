#SimGeom.py
import argparse
import json
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

# 尽量使用已有的配置与函数
from config.Geom_config import geom_args  # 已定义好的全局args
from dataset.Geom_dataset import GraphData
from model.GeomLayers.GeomModel import GeomModel
from utils.Geom_utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute similarity between two JSON graphs.")
    parser.add_argument('--json1', type=str, default="/home/vllm/encode/data/Geom/TEST_4096/QFN28LK(Cu)-90-450 Rev1_5.json", help='Path to the first JSON file')
    parser.add_argument('--json2', type=str,default="/home/vllm/encode/data/Geom/TEST_4096/QFN22LD(Cu) -532 Rev1_4.json", help='Path to the second JSON file')
    return parser.parse_args()


def load_single_graph_from_json(json_file: str) -> GraphData:
    """
    读取单个JSON图文件，构造GraphData对象。

    JSON格式需包含:
      {
        "n_num": int,
        "src": str,            # 图名称
        "features": [[...], ...],
        "succs": [[...], ...],
        "2D-index": [[x,y], ...]
      }
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        g_info = json.load(f)

    n_num = g_info['n_num']
    if n_num <= 0:
        raise ValueError(f"Graph n_num = {n_num} <= 0, invalid graph.")

    graph = GraphData(node_num=n_num, name=g_info.get('src', 'unknown'))

    # 加载特征
    if 'features' not in g_info:
        raise ValueError("JSON missing 'features' field.")
    graph.features = np.array(g_info['features'], dtype=np.float32)

    # 加载pos2d (可选)
    if '2D-index' in g_info:
        graph.pos2d = np.array(g_info['2D-index'], dtype=np.float32)

    # 添加边
    succs = g_info.get('succs', [])
    for u in range(n_num):
        if u < len(succs):
            for v in succs[u]:
                graph.add_edge(u, v)

    return graph


def process_single_graph_local(graph: GraphData, max_nodes: int, feature_dim: int):
    """
    对单个图进行对齐处理：
      - adjacency加自环
      - 特征 & pos2d 可能截断/补齐
      - 生成mask
    并将结果存放到 graph.matrices 中 (feature_matrix, adj_padded, mask, pos2d_matrix).
    """
    # 1) 邻接矩阵 (dense)，并加上自环
    adj = nx.adjacency_matrix(graph.adj).toarray()
    np.fill_diagonal(adj, 1)  # add self-loop

    actual_n = adj.shape[0]
    # 2) 若实际节点数>max_nodes，则截断邻接&特征&pos2d
    if actual_n > max_nodes:
        adj = adj[:max_nodes, :max_nodes]
        actual_n = max_nodes
        if graph.features.shape[0] > max_nodes:
            graph.features = graph.features[:max_nodes, :]
        if (graph.pos2d is not None) and (graph.pos2d.shape[0] > max_nodes):
            graph.pos2d = graph.pos2d[:max_nodes, :]

    # 3) 邻接矩阵补零
    adj_padded = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    adj_padded[:actual_n, :actual_n] = adj[:actual_n, :actual_n]

    # 4) 特征矩阵补 -1
    feature_matrix = np.full((max_nodes, feature_dim), -1, dtype=np.float32)
    if graph.features.ndim == 1:
        # 若只有一维，需要 reshape 为 [1, -1]
        graph.features = graph.features.reshape(1, -1)
    feature_matrix[:actual_n, :] = graph.features[:actual_n, :]

    # 5) mask
    mask = np.zeros((max_nodes,), dtype=np.float32)
    mask[:actual_n] = 1.0

    # 6) pos2d补 -1
    pos2d_matrix = np.full((max_nodes, 2), -1, dtype=np.float32)
    if graph.pos2d is not None:
        pos2d_matrix[:actual_n, :] = graph.pos2d[:actual_n, :]

    # 存储到 graph.matrices
    graph.matrices = (feature_matrix, adj_padded, mask, pos2d_matrix)


def prepare_single_graph_data(graph: GraphData, device: torch.device):
    """
    将 graph.matrices 中的数据转为 PyTorch tensor 并放到指定 device 上.
    返回 (features, adj, mask, pos2d).
    """
    if graph.matrices is None:
        raise ValueError("Graph has not been preprocessed. Call process_single_graph_local first.")

    feat, adj_mat, mask_mat, pos2d_mat = graph.matrices

    features = torch.tensor(feat, dtype=torch.float32, device=device)
    adj = torch.tensor(adj_mat, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_mat, dtype=torch.float32, device=device)
    pos2d = torch.tensor(pos2d_mat, dtype=torch.float32, device=device)

    return features, adj, mask, pos2d


def main():
    # 1) 解析命令行参数
    args_cli = parse_args()

    # 2) 可以根据需要修改 geom_args 的一些默认值，例如:
    # geom_args.gpu_index = '0'
    # ...

    # 固定随机种子 & 设备
    set_seed(geom_args.seed)
    device = get_device(geom_args)

    # 3) 分别读取 json1 与 json2
    g1 = load_single_graph_from_json(args_cli.json1)
    g2 = load_single_graph_from_json(args_cli.json2)

    # 4) 对单个图进行本地对齐处理(不调用 GeomDataset)
    max_nodes = geom_args.graph_size_max      # 默认4096
    feature_dim = geom_args.graph_init_dim    # 默认44

    process_single_graph_local(g1, max_nodes, feature_dim)
    process_single_graph_local(g2, max_nodes, feature_dim)

    # 5) 准备模型输入
    f1, a1, m1, p1 = prepare_single_graph_data(g1, device)
    f2, a2, m2, p2 = prepare_single_graph_data(g2, device)

    # 6) 加载已训练好的 GeomModel
    model = GeomModel(
        init_dim=feature_dim,
        d_model=32
    ).to(device)

    # 你的权重文件
    ckpt_path = "/home/vllm/encode/checkpoints/best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 7) 前向计算(不进行节点对齐)
    with torch.no_grad():
        # 注意：这里batch维度B=1，因此需要unsqueeze(0)
        x_enc1, _, _ = model(f1.unsqueeze(0), p1.unsqueeze(0), a1.unsqueeze(0), m1.unsqueeze(0))
        x_enc2, _, _ = model(f2.unsqueeze(0), p2.unsqueeze(0), a2.unsqueeze(0), m2.unsqueeze(0))

    # 8) 图级向量 (对64个节点做mean pooling)
    gvec1 = x_enc1.mean(dim=1)  # [1, d_model]
    gvec2 = x_enc2.mean(dim=1)  # [1, d_model]

    # 9) 计算余弦相似度 => 标量
    cos_val = F.cosine_similarity(gvec1, gvec2, dim=-1)  # shape [1]
    cos_val = cos_val.item()  # 取出浮点值

    # 将余弦相似度映射到 [0, 1]
    sim_01 = 0.5 * (cos_val + 1.0)

    # 10) 输出结果
    print(f"Cosine similarity (normalized to [0,1]): {sim_01:.6f}")


if __name__ == "__main__":
    main()
