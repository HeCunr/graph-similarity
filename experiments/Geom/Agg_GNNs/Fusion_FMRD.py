#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion_FMRD.py

本脚本实现以下流程：
1. 批量读取目录中的 JSON (几何信息) 和 H5 (序列信息) 文件；
2. 根据文件名匹配，将同名（去后缀）的一对 (json, h5) 送入已训练好的 Fusion 模型，得到预计算融合向量；
3. 构建所有文件的相似性矩阵（N x N）；
4. 读取 ground_truth.txt（其中每个小列表代表同一组），将同组文件视作“真实正例”；
5. 对每个文件，进行检索（选取相似度最高的 top_n 个文件，排除其自身），并计算 F1、AP、Recall、NDCG 四个指标；
6. 最终输出全局这四项指标的平均值。

评价指标参考了信息检索及度量学习的通行做法（如在 SIGIR、ICLR、CVPR 等会议论文中使用的 MAP、NDCG 等）。
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

# ------------------ 引入您已有的 SimFusion.py 中关键逻辑 ------------------
# 假设 SimFusion.py 中提供了若干可复用的函数:
#   1) load_fusion_args()
#   2) GraphMatchNetwork, MultiLevelNodeAggregator, SeqTransformer 等模型定义
#   3) load_checkpoint(geom_model, geom_agg, seq_model, ckpt_path, device)
#   4) parse_geom_json(json_path, max_nodes=4096, feat_dim=44)
#   5) parse_seq_h5(h5_path)
#   6) forward_geom_for_inference(...)
#   7) forward_seq_for_inference(...)
# 等等。请根据实际项目结构做相应修改。
from SimFusion import (
    load_fusion_args,
    GraphMatchNetwork,
    MultiLevelNodeAggregator,
    SeqTransformer,
    load_checkpoint,
    parse_geom_json,
    parse_seq_h5,
    forward_geom_for_inference,
    forward_seq_for_inference,
)
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Fusion-based FMRD evaluation for DXF files.")
    parser.add_argument("--geom_dir", type=str,
                        default="/home/vllm/MulConDXF/data/Geom/TEST_4096",
                        help="Directory containing JSON geometry files.")
    parser.add_argument("--seq_dir", type=str,
                        default="/home/vllm/MulConDXF/data/Seq/TEST_4096",
                        help="Directory containing H5 sequence files.")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/vllm/MulConDXF/checkpoints/fusion_best.pt",
                        help="Path to the trained fusion checkpoint.")
    parser.add_argument("--ground_truth", type=str,
                        default="/home/vllm/MulConDXF/data/ground_truth.txt",
                        help="Path to ground truth txt.")
    parser.add_argument("--top_n", type=int, default=10,
                        help="How many items to retrieve for evaluation (exclude itself).")
    parser.add_argument("--gpu_index", type=str, default='0',
                        help="Which GPU to use, default=0; use -1 for CPU.")
    return parser.parse_args()


def read_ground_truth(gt_path):
    """
    假设 ground_truth.txt 的内容类似:
    [
        ["QFN28LK(Cu)-90-450 Rev1_1",
         "QFN19LB(Cu) -503 Rev1_2",
         "QFN19LA(Cu) -502 Rev1_2",
         "QFN21LA(Cu) -508 Rev1_2",
         "QFN20LAH(Cu)-436 Rev1_2"],
        ["QFN28LK(Cu)-90-450 Rev1_2",
         "QFN21LA(Cu) -508 Rev1_3",
         ...
    ]
    每个小列表代表同一组文件。

    返回:
      group_dict: {filename -> group_id},
      groups: List of sets, 例如 [set1, set2, ...]
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # 解析可能是标准 json 数组的写法
    # ground_truth.txt 可能包含多行，需要保证能解析为有效 JSON
    # 如果有多行或注释，需要自行处理
    group_list = json.loads(content)

    group_dict = {}  # 每个文件 => 它所属的 group_id
    groups = []
    for idx, g in enumerate(group_list):
        # g 是一个列表，里面是同组的文件名
        file_set = set()
        for fn in g:
            # 去掉可能的多余空格后存入
            name_str = fn.strip()
            file_set.add(name_str)
            group_dict[name_str] = idx
        groups.append(file_set)
    return group_dict, groups


def compute_metrics_for_query(query_name, ranking_list, group_dict, groups, top_n):
    """
    对单个查询文件 query_name 计算 F1, AP, Recall, NDCG 等检索指标。

    参数:
      query_name   : 当前查询文件名 (不含后缀的部分或含后缀 - 依照 ground_truth.txt 的记录方式)
      ranking_list : 按相似度从高到低排序后的文件名列表（已排除了 query 自身）
      group_dict   : {filename -> group_id} 映射
      groups       : 每个组的 set
      top_n        : 取前 top_n 进行统计

    返回:
      f1, ap, recall, ndcg
    """

    # 1) 确定 query 所属的真实组
    #    同组中的文件(除自身)都视为 relevant
    if query_name not in group_dict:
        # 如果 ground_truth 中没有记录这个文件，默认本次检索无法计算，返回空
        return None, None, None, None

    query_gid = group_dict[query_name]
    relevant_set = groups[query_gid].copy()
    if query_name in relevant_set:
        relevant_set.remove(query_name)

    # 2) 取检索列表的 top_n
    retrieved_top_n = ranking_list[:top_n]

    # 3) 计算 若干指标

    # ---- (a) Precision, Recall, F1 ----
    retrieved_set = set(retrieved_top_n)
    tp = len(relevant_set.intersection(retrieved_set))
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    # ---- (b) Average Precision (AP) ----
    # AP 计算方法：对检索结果中每一个位置 i，若是 relevant，则计算 precision@i，然后求平均
    # ranking_list[:top_n] 中 relevant 的平均精确率
    hit_count = 0
    prec_sum = 0.0
    for i, fname in enumerate(retrieved_top_n, start=1):
        if fname in relevant_set:
            hit_count += 1
            prec_sum += (hit_count / i)
    ap = prec_sum / len(relevant_set) if len(relevant_set) > 0 else 0.0

    # ---- (c) NDCG ----
    # 对于二元相关性(相关/不相关)，DCG 计算中 relevant = 1, not relevant = 0
    # DCG = sum( (2^rel_i - 1) / log2(1+i) ), i 从 1 开始
    # rel_i = 1 if relevant, else 0
    # IDCG = DCG of ideal ranking => top(len(relevant_set)) all relevant
    def dcg_at_k(r_list):
        dcg_val = 0.0
        for i, rel in enumerate(r_list, start=1):
            # rel 要么 0/1，这里是二元
            if rel > 0:
                dcg_val += (1.0 / np.log2(i + 1))
        return dcg_val

    # 构建实际检索列表的 relevance
    rel_list = [1 if fname in relevant_set else 0 for fname in retrieved_top_n]
    dcg = dcg_at_k(rel_list)

    # 构建理想排名(全部相关的在前，如果真实相关数 < top_n，其余位置为 0)
    ideal_list = sorted(rel_list, reverse=True)
    idcg = dcg_at_k(ideal_list)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return f1, ap, recall, ndcg


def main():
    args = parse_args()

    # ------------------ 0) 设置设备与加载模型 ------------------
    base_args = load_fusion_args()
    base_args.gpu_index = args.gpu_index
    if torch.cuda.is_available() and int(base_args.gpu_index) >= 0:
        device = torch.device(f"cuda:{base_args.gpu_index}")
    else:
        device = torch.device("cpu")

    print(f"[Info] Using device: {device}")

    # 构建 几何模型 & 序列模型
    geom_model = GraphMatchNetwork(base_args.graph_init_dim, base_args).to(device)
    geom_agg   = MultiLevelNodeAggregator(in_features=base_args.graph_init_dim).to(device)
    seq_model  = SeqTransformer(
        d_model=base_args.d_model,
        num_layers=base_args.num_layers,
        nhead=base_args.nhead,
        dim_feedforward=base_args.dim_feedforward,
        dropout=base_args.dropout_seq,
        latent_dropout=0.1,
        use_selfatt_pool=True
    ).to(device)

    # 加载训练好的权重
    load_checkpoint(geom_model, geom_agg, seq_model, args.checkpoint, device)
    geom_model.eval()
    geom_agg.eval()
    seq_model.eval()

    # ------------------ 1) 遍历目录下的 JSON 和 H5 文件，并构建映射 ------------------
    #     假设 文件名形如 "xxx.json" / "xxx.h5", 我们用 xxx 作为 key。
    #     有些文件可能只在一个目录下出现、或者缺失，对缺失的文件无法计算向量，可进行跳过。
    geom_files = {}
    for file in os.listdir(args.geom_dir):
        if file.endswith(".json"):
            base_name = file[:-5]  # 去掉 .json 后缀
            geom_files[base_name] = os.path.join(args.geom_dir, file)

    seq_files = {}
    for file in os.listdir(args.seq_dir):
        if file.endswith(".h5"):
            base_name = file[:-3]  # 去掉 .h5 后缀
            seq_files[base_name] = os.path.join(args.seq_dir, file)

    # 找到两者都有的文件
    common_basenames = set(geom_files.keys()).intersection(set(seq_files.keys()))
    common_basenames = sorted(list(common_basenames))  # 为了固定顺序

    print(f"[Info] Found {len(common_basenames)} matched pairs of (json, h5).")

    # ------------------ 2) 对每个文件计算融合向量 ------------------
    fused_vecs = {}
    with torch.no_grad():
        for base_name in tqdm(common_basenames, desc="Computing fused vectors"):
            # 加载几何数据
            geom_feat, geom_adj, geom_mask = parse_geom_json(geom_files[base_name])
            # 加载序列数据
            seq_data = parse_seq_h5(seq_files[base_name])

            # 分别经过几何分支 & 序列分支
            geom_out = forward_geom_for_inference(geom_feat, geom_adj, geom_mask,
                                                  geom_model, geom_agg, device)
            seq_out  = forward_seq_for_inference(seq_data, seq_model, device)

            # 这里假设最终融合向量 = 0.5*(geom_out + seq_out)，并做一次归一化
            fused = F.normalize((geom_out + seq_out) * 0.5, dim=-1)

            fused_vecs[base_name] = fused.cpu().numpy()[0]  # shape [256], 存到 CPU

    # ------------------ 3) 构建相似性矩阵 ------------------
    #     对每个 base_name，计算与其它所有 base_name 的余弦相似度
    #     结果保存在 sim_matrix[i, j] 中
    N = len(common_basenames)
    sim_matrix = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        v1 = fused_vecs[common_basenames[i]]
        for j in range(N):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                v2 = fused_vecs[common_basenames[j]]
                # 余弦相似度
                score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                sim_matrix[i, j] = score

    # ------------------ 4) 读取 ground_truth.txt ------------------
    group_dict, groups = read_ground_truth(args.ground_truth)

    # ------------------ 5) 对每个文件进行检索评估，计算 F1, AP, Recall, NDCG ------------------
    #     对于索引 i 的文件，先对 sim_matrix[i,:] 做降序排序(排除自身)，
    #     然后对 top_n 做评估
    top_n = args.top_n
    f1_list = []
    ap_list = []
    recall_list = []
    ndcg_list = []

    for i in range(N):
        query_name = common_basenames[i]
        # 根据相似度排名，越大越相似
        row = sim_matrix[i, :]
        sort_idx = np.argsort(-row)  # 降序排序的索引
        ranking = []
        for idx_j in sort_idx:
            if idx_j == i:
                # 排除自身
                continue
            ranking.append(common_basenames[idx_j])

        # 计算该文件对应的检索指标
        f1, ap, r, ndcg = compute_metrics_for_query(query_name, ranking, group_dict, groups, top_n)
        if f1 is not None:
            f1_list.append(f1)
            ap_list.append(ap)
            recall_list.append(r)
            ndcg_list.append(ndcg)

    # ------------------ 6) 汇总并输出 ------------------
    if len(f1_list) == 0:
        print("[Warning] No valid queries were found in ground_truth. Check file naming or ground_truth content.")
        return

    mean_f1     = float(np.mean(f1_list))
    mean_ap     = float(np.mean(ap_list))
    mean_recall = float(np.mean(recall_list))
    mean_ndcg   = float(np.mean(ndcg_list))

    print("========================================================")
    print(f"[Final Result] top_n={top_n}")
    print(f"  F1     = {mean_f1:.4f}")
    print(f"  AP     = {mean_ap:.4f}")
    print(f"  Recall = {mean_recall:.4f}")
    print(f"  NDCG   = {mean_ndcg:.4f}")
    print("========================================================")


if __name__ == "__main__":
    main()
