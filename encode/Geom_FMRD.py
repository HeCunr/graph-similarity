#Geom_FMRD.py
import os
import json
import math
import torch
import torch.nn.functional as F
import numpy as np

# 假设和 SimGeom.py 同级目录
import SimGeom

###############################################################################
# 1) 从 Geom_truth.json 加载 ground_truth, 生成 {文件名: 同组文件集合}
###############################################################################
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

###############################################################################
# 2) 预计算所有文件的图向量，并构造相似度矩阵(或保存在字典里)
###############################################################################
def compute_all_embeddings(file_list, simgeom_args, model, pooling_module, device):
    """
    读取 file_list 中的每个文件 (假设都是 .json)，
    调用 SimGeom.load_graph_from_json + SimGeom.get_graph_embedding 得到其 embedding。
    最终返回 dict: { filename: torch.Tensor([dim]) }
    """
    emb_dict = {}
    model.eval()
    pooling_module.eval()

    for fname in file_list:
        # 读取图数据 -> feat, adj, mask
        feat, adj, mask = SimGeom.load_graph_from_json(
            fname,
            max_n=4096,  # 或你需要的尺寸
            feat_dim=simgeom_args.graph_init_dim
        )
        # 得到图 embedding
        with torch.no_grad():
            emb = SimGeom.get_graph_embedding(feat, adj, mask, pooling_module, model, device)
        emb_dict[fname] = emb.cpu()  # 也可保留在GPU上，但会占用更多显存

    return emb_dict

def build_similarity_matrix(emb_dict):
    """
    给定 {filename: embedding}, 构造一个相似度矩阵 (或者直接构造 pairwise 相似度 map)。
    返回 sim_matrix 以及行列索引与文件名的映射。
    注意：相似度这里示例用 余弦相似度。
    """
    file_list = sorted(emb_dict.keys())
    n = len(file_list)
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    # 将所有 embedding 堆到一起做一次性矩阵乘法也可以
    # 这里演示简单两重循环
    for i in range(n):
        emb_i = emb_dict[file_list[i]]
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                emb_j = emb_dict[file_list[j]]
                # 计算 cos sim
                # 余弦相似度 = emb_i dot emb_j / (||emb_i|| * ||emb_j||)
                # 这里 emb_i, emb_j 已经是 normalize 后的，也可再做 F.cosine_similarity
                sim_val = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0), dim=-1).item()
                sim_matrix[i, j] = sim_val
    return file_list, sim_matrix

###############################################################################
# 3) find_top_similar_files: 利用预先算好的相似度矩阵做快速检索
###############################################################################
def find_top_similar_files(query_file, file_list, sim_matrix, top_n=10):
    """
    输入:
      query_file: 目标文件名
      file_list:  与 sim_matrix 对应的文件名列表
      sim_matrix: NxN 相似度矩阵 (按 file_list 排序)
      top_n:      需要返回的数量

    输出:
      [(fname, sim), ...] 前 top_n 个(相似度最高)
      （不包含 query_file 本身）
    """
    # 找到 query_file 在 file_list 中的索引
    if query_file not in file_list:
        raise ValueError(f"Query file {query_file} not found in file_list.")
    qidx = file_list.index(query_file)

    # 取出该行
    row = sim_matrix[qidx, :]  # shape = [N]
    # 排序：相似度从大到小
    indices_sorted = np.argsort(-row)  # 降序
    results = []
    for idx in indices_sorted:
        if file_list[idx] == query_file:
            continue  # 跳过自己
        results.append((file_list[idx], float(row[idx])))

    return results[:top_n]

###############################################################################
# 4) 度量指标：Precision/Recall/F1, AP, NDCG
###############################################################################
def precision_recall_f1(recommended_list, ground_truth_set):
    """
    recommended_list: 推荐的文件名列表 (不含 query_file 本身).
    ground_truth_set: 真实同组集合(包含 query_file 本身)，需先排除自己。
    返回 (precision, recall, f1).
    """
    # 去掉自己
    # 如果 ground_truth_set 包含 query_file，也可以 ground_truth_set - {query_file}
    # 这里简写:
    #   先复制
    gt_set = set(ground_truth_set)
    # (本示例假定 ground_truth_set 里一定有自己)
    #   实际可做更安全的 discard
    recommended_set = set(recommended_list)

    # 命中数量
    hits = recommended_set.intersection(gt_set)
    precision = len(hits) / len(recommended_list) if recommended_list else 0.0
    recall = len(hits) / (len(gt_set) if len(gt_set) else 1e-9)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def average_precision(recommended_list, ground_truth_set):
    """
    计算 AP (Average Precision).
    当 ground_truth_set 为空时，AP=0.
    """
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
    计算 NDCG, 二元相关(文件在 ground_truth_set 中 -> relevant)。
    k: 只计算前 k 个；若不指定则默认 len(recommended_list).
    """
    if k is None or k > len(recommended_list):
        k = len(recommended_list)

    dcg = 0.0
    for i in range(k):
        rel = 1.0 if recommended_list[i] in ground_truth_set else 0.0
        dcg += (2**rel - 1) / math.log2(i + 2)  # i从0开始 => log2(i+2)

    # 计算 ideal DCG
    # 先看 ground_truth 有多少 relevant
    G = len(ground_truth_set)
    ideal_hits = min(G, k)
    idcg = 0.0
    for i in range(ideal_hits):
        rel = 1.0
        idcg += (2**rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

###############################################################################
# 5) 主函数：流程串起来
###############################################################################
def main():
    # ============ 读取命令行参数(或直接使用 SimGeom.parse_args) ============
    args = SimGeom.parse_args()
    device = SimGeom.get_device(args)

    # ============ (1) 加载 ground_truth ============
    ground_truth_path = "/home/vllm/encode/data/Geom/Geom_truth.json"  # 根据实际路径
    file2group = load_ground_truth(ground_truth_path)

    # ============ (2) 收集需要计算的文件(示例中为同目录下 .json 文件) ============
    #   假设文件位于 "/home/vllm/encode/data/Geom/TEST_4096"
    data_dir = "/home/vllm/encode/data/Geom/TEST_4096"
    all_json_files = []
    for f in os.listdir(data_dir):
        if f.endswith(".json"):
            # 拼出完整路径
            full_path = os.path.join(data_dir, f)
            all_json_files.append(full_path)

    # 根据 ground_truth 中的文件名格式，需要将 "QFN28LK(Cu)-90-450 Rev1_5.json" 拼在一起
    # 这里仅示例：full_path 末尾文件名 => ground_truth 里只存文件名 => 后面要对照一下
    #   你可以把 all_json_files 改成存储纯文件名，然后在 load_graph_from_json 时再拼 data_dir
    #   视你项目结构而定
    #
    # 这里为了指标计算的一致性，要与 ground_truth 里的 key 对应
    # ground_truth 里的 key = "QFN28LK(Cu)-90-450 Rev1_5.json" (示例)
    #
    # 下面简单地把全部文件名拆出来做一个映射
    name_map = {}
    for f in all_json_files:
        basename = os.path.basename(f)
        name_map[basename] = f

    # 只计算 ground_truth 字典里出现过的文件(交集)，或使用 all_json_files
    # 下面以 ground_truth 为准
    target_files = list(file2group.keys())
    # 取有实际文件的那部分
    actual_files = [f for f in target_files if f in name_map]
    actual_files_paths = [name_map[f] for f in actual_files]

    # ============ (3) 初始化模型 & pooling_module, 并加载权重 ============
    checkpoint = torch.load(args.checkpoint, map_location=device)
    pooling_module = SimGeom.MultiLevelNodeAggregator(in_features=args.graph_init_dim).to(device)
    model = SimGeom.GraphMatchNetwork(node_init_dims=args.graph_init_dim, args=args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'pooling_module_state_dict' in checkpoint:
        pooling_module.load_state_dict(checkpoint['pooling_module_state_dict'])
    else:
        raise ValueError("Checkpoint does not contain pooling_module_state_dict.")

    # ============ (4) 预计算所有文件的 embeddings ============
    emb_dict = compute_all_embeddings(actual_files_paths, args, model, pooling_module, device)

    # 这里 emb_dict 的 key 是 "完整路径"；而 ground_truth 用的是文件名
    # 可以把 key 改回纯文件名(以便后面使用)
    pure_emb_dict = {}
    for fname_path, emb in emb_dict.items():
        bname = os.path.basename(fname_path)
        pure_emb_dict[bname] = emb

    # ============ (5) 构建相似度矩阵，或使用 pairwise 计算也可 ============
    file_list, sim_matrix = build_similarity_matrix(pure_emb_dict)

    # ============ (6) 对每个文件，找 top_n 推荐，并计算指标 ============
    top_n = 10
    all_f1s = []
    all_aps = []
    all_recalls = []
    all_ndcgs = []

    for query_file in file_list:
        # 推荐
        top_sim = find_top_similar_files(query_file, file_list, sim_matrix, top_n=top_n)
        recommended_list = [item[0] for item in top_sim]  # 拿文件名

        # ground truth
        if query_file not in file2group:
            # 如果 ground_truth 里没有，就跳过
            continue
        gt_set = set(file2group[query_file])
        # 需要排除自身
        if query_file in gt_set:
            gt_set.remove(query_file)

        # 计算指标
        precision, recall, f1 = precision_recall_f1(recommended_list, gt_set)
        ap = average_precision(recommended_list, gt_set)
        ndcg = ndcg_k(recommended_list, gt_set, k=top_n)

        all_f1s.append(f1)
        all_aps.append(ap)
        all_recalls.append(recall)
        all_ndcgs.append(ndcg)

    # ============ (7) 汇总均值输出 ============
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
