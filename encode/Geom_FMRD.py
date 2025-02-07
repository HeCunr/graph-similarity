import os
import json
import math
import torch
import torch.nn.functional as F
import numpy as np

# ===== 导入你项目中已有的脚本和函数 =====
from config.Geom_config import geom_args
from model.GeomLayers.GeomModel import GeomModel

from SimGeom import (
    load_single_graph_from_json,
    process_single_graph_local,
    prepare_single_graph_data,
)

# -------------------------------------------------------------------
# 1) 加载 ground_truth: {文件名 => 同组文件集合}
# -------------------------------------------------------------------
def load_ground_truth(gt_json_path: str):
    """
    给定 Geom_truth.json 的路径，如:
    /home/vllm/encode/data/Geom/Geom_truth.json

    其内容格式类似:
    [
      ["fileA.json", "fileB.json", "fileC.json"],
      ["fileX.json", "fileY.json"],
      ...
    ]

    返回 {文件名: set([...])}，使得同组的文件彼此都在同一个 set 里。
    """
    if not os.path.exists(gt_json_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_json_path}")

    with open(gt_json_path, 'r', encoding='utf-8') as f:
        group_list = json.load(f)  # 读取为 list of list

    file2group = {}
    for group in group_list:
        gset = set(group)
        for fname in group:
            # 如果某文件已出现过，可做 union 合并
            if fname not in file2group:
                file2group[fname] = set()
            file2group[fname] = file2group[fname].union(gset)

    return file2group

# -------------------------------------------------------------------
# 2) 单文件 -> 图向量（使用与 SimGeom.py 相同的逻辑）
# -------------------------------------------------------------------
def compute_single_file_embedding(json_path: str, model: GeomModel, device: torch.device):
    """
    读取单个 .json 文件，并使用 GeomModel（无对齐）输出图级向量(64个节点均值)。
    """
    # 1) Load JSON -> GraphData
    graph = load_single_graph_from_json(json_path)

    # 2) Process local => padding/cropping
    max_nodes = geom_args.graph_size_max
    feature_dim = geom_args.graph_init_dim
    process_single_graph_local(graph, max_nodes, feature_dim)

    # 3) 准备输入 (tensor on device)
    f, a, m, p = prepare_single_graph_data(graph, device)

    # 4) 前向计算 => node embeddings => mean pool => [1, d_model]
    with torch.no_grad():
        x_enc, _, _ = model(f.unsqueeze(0), p.unsqueeze(0), a.unsqueeze(0), m.unsqueeze(0))
        # mean pooling => [1, d_model]
        gvec = x_enc.mean(dim=1)  # shape [1, d_model]
    return gvec.squeeze(0)  # => [d_model]

# -------------------------------------------------------------------
# 3) 构建相似度矩阵
# -------------------------------------------------------------------
def build_similarity_matrix(emb_dict):
    """
    给定 {filename: embedding (d_model, )}, 构造一个相似度矩阵 (NxN)。
    返回 (file_list, sim_matrix)。
    """
    file_list = sorted(emb_dict.keys())
    n = len(file_list)
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        emb_i = emb_dict[file_list[i]]  # shape [d_model]
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                emb_j = emb_dict[file_list[j]]
                sim_val = F.cosine_similarity(
                    emb_i.unsqueeze(0), emb_j.unsqueeze(0), dim=-1
                ).item()
                sim_matrix[i, j] = sim_val
    return file_list, sim_matrix

# -------------------------------------------------------------------
# 4) 检索 top_k 并评估
# -------------------------------------------------------------------
def find_top_similar_files(query_file, file_list, sim_matrix, top_n=10):
    """
    输入:
      query_file: 目标文件名
      file_list:  与 sim_matrix 对应的文件名列表(同顺序)
      sim_matrix: NxN 相似度矩阵
      top_n:      返回相似度最高的数量(排除自身)

    输出:
      [(fname, sim), ...] 前 top_n 个(相似度最高)
    """
    if query_file not in file_list:
        raise ValueError(f"Query file {query_file} not found in file_list.")
    qidx = file_list.index(query_file)

    row = sim_matrix[qidx]  # shape [N]
    # 降序排序
    indices_sorted = np.argsort(-row)
    results = []
    for idx in indices_sorted:
        if file_list[idx] == query_file:
            continue
        results.append((file_list[idx], float(row[idx])))
    return results[:top_n]

# ============ 度量指标 ============

def precision_recall_f1(recommended_list, ground_truth_set):
    """
    recommended_list: 推荐的文件名列表 (不含 query_file 本身).
    ground_truth_set: 同组集合(包含 query_file), 使用时需排除 query_file.
    """
    recommended_set = set(recommended_list)
    # 去掉自己
    # ground_truth_set 里可能包含 query_file，自行排除
    gt_set = set(ground_truth_set)
    # 若 query_file 在内，则剔除
    #   (在外部剔除也行，这里做一下安全处理)
    #   gt_set.discard(query_file)

    hits = recommended_set.intersection(gt_set)
    precision = len(hits) / (len(recommended_list) if recommended_list else 1e-9)
    recall = len(hits) / (len(gt_set) if gt_set else 1e-9)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def average_precision(recommended_list, ground_truth_set):
    """计算 AP (Average Precision). 二元相关模式"""
    gt_set = set(ground_truth_set)
    if not gt_set:
        return 0.0
    hit_count = 0
    sum_precisions = 0.0
    for i, fname in enumerate(recommended_list):
        if fname in gt_set:
            hit_count += 1
            precision_at_i = hit_count / (i + 1)
            sum_precisions += precision_at_i
    return sum_precisions / len(gt_set)

def ndcg_k(recommended_list, ground_truth_set, k=None):
    """
    计算 NDCG, 二元相关(文件在 ground_truth_set 中 => relevant)。
    k: 只计算前 k 个；若不指定则默认 len(recommended_list).
    """
    if (k is None) or (k > len(recommended_list)):
        k = len(recommended_list)

    dcg = 0.0
    for i in range(k):
        rel = 1.0 if recommended_list[i] in ground_truth_set else 0.0
        dcg += (2**rel - 1) / math.log2(i + 2)

    # ideal DCG
    G = len(ground_truth_set)
    ideal_hits = min(G, k)
    idcg = 0.0
    for i in range(ideal_hits):
        rel = 1.0
        idcg += (2**rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

# -------------------------------------------------------------------
# 5) 主函数
# -------------------------------------------------------------------
def main():
    # ============ 读取 ground truth ============
    ground_truth_path = "/home/vllm/encode/data/Geom/Geom_truth.json"  # 请根据实际路径修改
    file2group = load_ground_truth(ground_truth_path)

    # ============ 收集需要处理的文件（测试集 / 全部 .json）============
    #   举例：测试文件位于 "/home/vllm/encode/data/Geom/TEST_4096"
    data_dir = "/home/vllm/encode/data/Geom/TEST_4096"
    all_json_files = []
    for fn in os.listdir(data_dir):
        if fn.endswith(".json"):
            full_path = os.path.join(data_dir, fn)
            all_json_files.append(full_path)

    # ground_truth 中保存的是纯文件名，比如 "QFN28LK(Cu)-90-450 Rev1_5.json"
    # 因此我们需要一份 {basename => full_path} 的映射
    name2path = {os.path.basename(p): p for p in all_json_files}

    # 与 ground_truth 相交的文件名
    candidate_files = list(file2group.keys())  # ground_truth 中的文件名
    actual_files = [f for f in candidate_files if f in name2path]

    if not actual_files:
        print("No intersection between ground_truth and actual JSON files. Nothing to do.")
        return

    # ============ 加载已训练好的 GeomModel ============
    device = torch.device("cuda" if torch.cuda.is_available() and int(geom_args.gpu_index) >= 0 else "cpu")
    model = GeomModel(
        init_dim=geom_args.graph_init_dim,  # 44
        d_model=32
    ).to(device)

    ckpt_path = "/home/vllm/encode/checkpoints/best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ============ 逐文件计算 embedding ============
    emb_dict = {}
    for fname in actual_files:
        json_path = name2path[fname]
        emb = compute_single_file_embedding(json_path, model, device)
        emb_dict[fname] = emb.cpu()  # 存在 CPU 上，防止显存累积

    # ============ 构建相似度矩阵 ============
    file_list, sim_matrix = build_similarity_matrix(emb_dict)

    # ============ 对每个文件做检索 & 计算评估指标 ============
    top_n = 10
    all_f1s = []
    all_aps = []
    all_recalls = []
    all_ndcgs = []

    for query_file in file_list:
        # 找到最相似的若干文件
        top_sim = find_top_similar_files(query_file, file_list, sim_matrix, top_n=top_n)
        recommended_list = [item[0] for item in top_sim]

        # ground_truth
        if query_file not in file2group:
            # ground_truth 中可能没有
            continue
        gt_set = set(file2group[query_file])
        # 排除 query_file 本身
        gt_set.discard(query_file)

        # 计算指标
        precision, recall, f1 = precision_recall_f1(recommended_list, gt_set)
        ap = average_precision(recommended_list, gt_set)
        ndcg_val = ndcg_k(recommended_list, gt_set, k=top_n)

        all_f1s.append(f1)
        all_aps.append(ap)
        all_recalls.append(recall)
        all_ndcgs.append(ndcg_val)

    # ============ 汇总输出 ============
    mean_f1 = np.mean(all_f1s) if all_f1s else 0.0
    mean_ap = np.mean(all_aps) if all_aps else 0.0
    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    mean_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0

    print("===== Final Metrics =====")
    print(f"Mean F1:       {mean_f1:.4f}")
    print(f"Mean AP:       {mean_ap:.4f}")
    print(f"Mean Recall:   {mean_recall:.4f}")
    print(f"Mean NDCG:     {mean_ndcg:.4f}")

if __name__ == "__main__":
    main()
