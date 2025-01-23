#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepDXF_FMRD.py
基于已训练好的DeepDXF Transformer模型，对 .h5 文件进行相似度计算(余弦相似度，归一化到[0,1])，
并在此基础上评估 F1, MAP, Recall, NDCG。

【GROUND TRUTH 说明】
- ground_truth 文件是一个 JSON，形如：
  [
    ["fileA.h5", "fileB.h5", ...],
    ["fileC.h5", "fileD.h5", ...],
    ...
  ]
  若 fileA.h5 位于第一个group，表示它的 ground_truth_list 就是同组的所有文件 ["fileA.h5", "fileB.h5", ...]。

【使用方法示例】
  python DeepDXF_FMRD.py \
      --ground_truth /path/to/DeepDXF_truth.json \
      --folder_path /path/to/h5_folder \
      --model_path /path/to/best_model.pth \
      --top_n 20
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py

# ====================== 1) SimDeepDXF 类：仅保留骨干网络做相似度计算 ======================
from model.transformer_encoder import DXFTransformer
import torch.nn as nn

class DXFBackbone(nn.Module):
    """
    从 DXFTransformer 中只保留 embedding, progressive_pool, encoder 三部分，
    丢弃 projection/mlp, BN, dropout。仅用于抽取表征。
    """
    def __init__(self, full_model: DXFTransformer):
        super().__init__()
        self.embedding = full_model.embedding
        self.progressive_pool = full_model.progressive_pool
        self.encoder = full_model.encoder

    @torch.no_grad()
    def forward(self, entity_type, entity_params):
        # entity_type: (batch, 4096), entity_params: (batch, 4096, 43)
        src = self.embedding(entity_type, entity_params)         # => (B,4096,256)
        src = self.progressive_pool(src)                         # => (B,64,256)
        src = src.permute(1, 0, 2)                               # => (64,B,256)
        memory = self.encoder(src)                               # => (64,B,256)
        memory = memory.permute(1, 0, 2)                         # => (B,64,256)
        return memory

class SimDeepDXF:
    """
    使用已经训练好的 DXFTransformer 权重，构建 backbone 进行表征提取，
    然后计算两个 .h5 文件的余弦相似度（并映射到 [0,1]）。
    """
    def __init__(self, model_path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 加载完整模型
        from model.transformer_encoder import DXFTransformer
        full_model = DXFTransformer(
            d_model=256,
            num_layers=6,
            dim_z=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            latent_dropout=0.3
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        full_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        full_model.eval().to(self.device)

        # 构建只保留骨干的网络
        self.backbone = DXFBackbone(full_model).to(self.device)
        self.backbone.eval()

    @torch.no_grad()
    def extract_file_representation(self, h5_path):
        """
        提取给定 h5 文件的“整体表征向量”(维度256)，
        如果文件包含多条记录，就将它们的均值作为文件表征。
        """
        with h5py.File(h5_path, 'r') as f:
            if 'dxf_vec' not in f:
                print(f"[Warning] {h5_path} 中无 'dxf_vec' 数据集.")
                return None
            dset = f['dxf_vec']
            n_samples = len(dset)

        if n_samples < 1:
            print(f"[Warning] {h5_path} 文件里无有效样本.")
            return None

        all_vecs = []
        for i in range(n_samples):
            with h5py.File(h5_path, 'r') as f:
                data_i = f['dxf_vec'][i]  # shape=(4096,44)

            entity_type = torch.tensor(data_i[:, 0], dtype=torch.long, device=self.device).unsqueeze(0)
            entity_params = torch.tensor(data_i[:, 1:], dtype=torch.float, device=self.device).unsqueeze(0)
            # => (1,4096), (1,4096,43)

            # 前向
            memory = self.backbone(entity_type, entity_params)  # (1,64,256)
            memory_mean = memory.mean(dim=1)                    # => (1,256)

            all_vecs.append(memory_mean.cpu().numpy())          # => (1,256)

        # 拼为 (n_samples,256)，再做平均 => (256,)
        all_vecs = np.concatenate(all_vecs, axis=0)  # => (n_samples,256)
        file_vec = all_vecs.mean(axis=0)             # => (256,)
        return file_vec

    @torch.no_grad()
    def compare_h5_files(self, h5_fileA, h5_fileB):
        """
        计算两个h5文件的相似度 [0,1]
        若有错误或无法计算，则返回 None
        """
        vecA = self.extract_file_representation(h5_fileA)
        vecB = self.extract_file_representation(h5_fileB)
        if vecA is None or vecB is None:
            return None

        a_t = torch.tensor(vecA, dtype=torch.float, device=self.device)
        b_t = torch.tensor(vecB, dtype=torch.float, device=self.device)
        a_norm = F.normalize(a_t, dim=0)
        b_norm = F.normalize(b_t, dim=0)
        cos_sim = torch.dot(a_norm, b_norm).item()      # [-1,1]
        sim_01 = (cos_sim + 1.0) / 2.0                  # [0,1]
        return sim_01


# ====================== 2) 评估指标 + 主流程：计算F1, MAP, Recall, NDCG ======================

def load_ground_truth(ground_truth_path):
    """
    加载 ground truth JSON (list of groups),
    返回字典: { filename: [同组其他文件们], ...}
    """
    with open(ground_truth_path, 'r') as f:
        data = json.load(f)
    ground_truth = {}
    for group in data:
        for file in group:
            ground_truth[file] = group
    return ground_truth


def calculate_fbeta_score(recommended_list, true_list, beta=1.0):
    rec_set = set(recommended_list)
    true_set = set(true_list)
    tp = len(rec_set & true_set)
    fp = len(rec_set - true_set)
    fn = len(true_set - rec_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    beta_squ = beta ** 2

    denom = (beta_squ * precision) + recall
    if denom == 0:
        return 0.0
    f_beta = (1 + beta_squ) * (precision * recall) / denom
    return f_beta

def average_precision(recommended_list, true_list):
    hits = 0
    precision_sum = 0.0
    true_size = len(true_list)
    for i, item in enumerate(recommended_list):
        if item in true_list:
            hits += 1
            precision_sum += hits / (i + 1)
    if true_size == 0:
        return 0.0
    return precision_sum / true_size

def calculate_recall(recommended_list, true_list):
    rec_set = set(recommended_list)
    true_set = set(true_list)
    tp = len(rec_set & true_set)
    fn = len(true_set - rec_set)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall

def calculate_ndcg(recommended_list, true_list, top_n):
    def dcg(scores):
        return sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores))

    # relevance in top_n
    rel_scores = [1 if x in true_list else 0 for x in recommended_list[:top_n]]
    dcg_val = dcg(rel_scores)

    ideal_len = min(len(true_list), top_n)
    ideal_scores = [1] * ideal_len
    ideal_dcg_val = dcg(ideal_scores)
    if ideal_dcg_val == 0:
        return 0.0
    return dcg_val / ideal_dcg_val

def find_top_similar_files(sim_dxf, target_file, folder_path, top_n):
    """
    对folder_path内其他所有 .h5 文件计算相似度，返回相似度最高的 top_n 个
    return: [(filepath, similarity), ...] (sorted)
    """
    similarity_list = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            candidate_path = os.path.join(root, fname)
            if candidate_path == target_file:
                continue
            sim = sim_dxf.compare_h5_files(target_file, candidate_path)
            if sim is not None:
                similarity_list.append((candidate_path, sim))

    # 排序 & 取 top_n
    sorted_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
    return sorted_list[:top_n]


def main():
    parser = argparse.ArgumentParser("DeepDXF_FMRD: F1, MAP, Recall, NDCG in one script with SimDeepDXF included")
    parser.add_argument("--ground_truth", default=r"/home/vllm/encode/data/DeepDXF/DeepDXF_truth.json", help="Path to DeepDXF_truth.json")
    parser.add_argument("--folder_path", default=r"/home/vllm/encode/data/DeepDXF/TEST_4096", help="Folder containing .h5 files")
    parser.add_argument("--model_path", default="/home/vllm/encode/pretrained/DeepDXF/DeepDXF.pth", help="Trained model checkpoint path")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top similar files to retrieve")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID to use")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1) 加载 ground truth
    ground_truth = load_ground_truth(args.ground_truth)

    # 2) 初始化相似度计算器
    sim_dxf = SimDeepDXF(args.model_path, device=device)

    # 3) 评估
    F1_list, AP_list, Recall_list, NDCG_list = [], [], [], []

    for root, _, files in os.walk(args.folder_path):
        for fname in files:
            # 如果该文件名不在 ground_truth 里，跳过
            if fname not in ground_truth:
                continue

            # 目标文件全路径
            target_path = os.path.join(root, fname)

            # 找 top_n 相似
            top_similar = find_top_similar_files(sim_dxf, target_path, args.folder_path, args.top_n)
            # 只取文件名
            recommended_list = [os.path.basename(p) for p, _ in top_similar]

            # ground truth list
            true_list = ground_truth[fname]

            # 计算 F1, AP, Recall, NDCG
            f1 = calculate_fbeta_score(recommended_list, true_list, beta=1.0)
            ap = average_precision(recommended_list, true_list)
            rec = calculate_recall(recommended_list, true_list)
            ndcg = calculate_ndcg(recommended_list, true_list, args.top_n)

            F1_list.append(f1)
            AP_list.append(ap)
            Recall_list.append(rec)
            NDCG_list.append(ndcg)

            print(f"[File: {fname}]")
            print(f"   F1@{args.top_n}     = {f1:.4f}")
            print(f"   AP@{args.top_n}     = {ap:.4f}")
            print(f"   Recall@{args.top_n} = {rec:.4f}")
            print(f"   NDCG@{args.top_n}   = {ndcg:.4f}")

    # 4) 统计均值
    mean_f1 = float(np.mean(F1_list)) if F1_list else 0.0
    mean_ap = float(np.mean(AP_list)) if AP_list else 0.0
    mean_recall = float(np.mean(Recall_list)) if Recall_list else 0.0
    mean_ndcg = float(np.mean(NDCG_list)) if NDCG_list else 0.0

    print("\n===== SUMMARY =====")
    print(f"Mean F1       = {mean_f1:.4f}")
    print(f"MAP           = {mean_ap:.4f}")
    print(f"Mean Recall   = {mean_recall:.4f}")
    print(f"Mean NDCG     = {mean_ndcg:.4f}")


if __name__ == "__main__":
    main()
