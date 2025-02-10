#dataset/Geom_toVec.py
import os
import json
import numpy as np
import networkx as nx
import h5py

from config.Geom_config import geom_args  # 若不想依赖geom_args，可自行定义参数
# 这里假设:
#   geom_args.data_dir => '/home/vllm/encode/data/Geom/TRAIN_4096'
#   geom_args.graph_size_max => 4096
#   geom_args.graph_init_dim => 44

def _build_adjacency_matrix(n_num, succs):
    """
    将 JSON 中的 'succs' 转为邻接矩阵，并加上 self-loop。
    succs: 类似 [[1,2], [2], [0], ...] 的列表结构
    """
    adj = np.zeros((n_num, n_num), dtype=np.float32)
    for u in range(n_num):
        if u < len(succs):
            for v in succs[u]:
                if 0 <= v < n_num:
                    adj[u, v] = 1.0
    # 对角线加上自环
    np.fill_diagonal(adj, 1.0)
    return adj

def _process_single_graph(g_info, max_n, feature_dim):
    """
    将单个 JSON 字典（g_info）处理成对齐后的 (feat_mat, adj_mat, mask_vec, pos2d_mat)，
    并返回 src, n_num 用于后续保存。
    """
    n_num = g_info.get('n_num', 0)
    src = g_info.get('src', "unknown_src")

    # 邻接矩阵
    succs = g_info.get('succs', [])
    adj = _build_adjacency_matrix(n_num, succs)

    # 特征
    features = g_info.get('features', [])
    features = np.array(features, dtype=np.float32)
    # 若只有一维，需要 reshape
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.shape[0] != n_num:
        # 如果尺寸对不上，可以根据情况处理或直接跳过
        pass

    # pos2d
    pos2d = g_info.get('2D-index', [])
    pos2d = np.array(pos2d, dtype=np.float32)
    if pos2d.ndim == 1:
        pos2d = pos2d.reshape(1, -1)

    # ----------- 截断/补齐 -----------
    actual_n = n_num
    if actual_n > max_n:
        # 截断
        adj = adj[:max_n, :max_n]
        features = features[:max_n, :]
        if pos2d.shape[0] >= max_n:
            pos2d = pos2d[:max_n, :]
        actual_n = max_n

    # 邻接矩阵补0
    adj_padded = np.zeros((max_n, max_n), dtype=np.float32)
    adj_padded[:actual_n, :actual_n] = adj[:actual_n, :actual_n]

    # 特征矩阵补 -1
    feature_matrix = np.full((max_n, feature_dim), -1, dtype=np.float32)
    row_feat = min(actual_n, features.shape[0])
    feature_matrix[:row_feat, :] = features[:row_feat, :]

    # mask
    mask = np.zeros((max_n,), dtype=np.float32)
    mask[:actual_n] = 1.0

    # pos2d 补 -1
    pos2d_matrix = np.full((max_n, 2), -1, dtype=np.float32)
    row_pos = min(actual_n, pos2d.shape[0])
    pos2d_matrix[:row_pos, :] = pos2d[:row_pos, :]

    return feature_matrix, adj_padded, mask, pos2d_matrix, src, n_num

def convert_json_to_h5_onefile(json_path, args):
    """
    将单个 json_path 文件（其中只包含一个图的 JSON 数据）处理并保存为同名 h5。
    如 "xxx.json" -> "xxx.h5"
    """
    if not json_path.endswith('.json'):
        return

    # 读取 json
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) == 0:
        print(f"[WARN] {json_path} is empty, skip.")
        return

    # 假设只有一行 JSON => 只对应一个图
    g_info = json.loads(lines[0].strip())
    n_num = g_info.get('n_num', 0)
    if n_num <= 0:
        print(f"[WARN] n_num <= 0 in {json_path}, skip.")
        return

    max_n = args.graph_size_max
    feature_dim = args.graph_init_dim
    feat_mat, adj_mat, mask_vec, pos2d_mat, src_str, node_num = _process_single_graph(g_info, max_n, feature_dim)

    # 构造输出 h5 路径
    h5_path = json_path[:-5] + '.h5'  # 把 ".json" 替换为 ".h5"
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset('src', data=src_str)
        h5f.create_dataset('n_num', data=node_num)
        h5f.create_dataset('feature_matrix', data=feat_mat)
        h5f.create_dataset('adj_padded', data=adj_mat)
        h5f.create_dataset('mask', data=mask_vec)
        h5f.create_dataset('pos2d_matrix', data=pos2d_mat)

    print(f"[OK] Wrote 1 graph => {h5_path}")

def main():
    data_dir = geom_args.data_dir  # '/home/vllm/encode/data/Geom/TRAIN_4096'
    all_files = os.listdir(data_dir)
    json_files = [f for f in all_files if f.endswith('.json')]

    for jf in json_files:
        json_path = os.path.join(data_dir, jf)
        convert_json_to_h5_onefile(json_path, geom_args)

    print(f"Done. Processed {len(json_files)} json files => .h5")

if __name__ == "__main__":
    main()
