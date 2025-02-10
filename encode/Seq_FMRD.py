# Seq_FMRD.py
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 从 SimSeq.py 中导入刚才定义的 SimSeq 类
from SimSeq import SimSeq

# ====================== 评估指标计算 ======================
def calculate_fbeta_score(recommended_list, true_list, beta=1.0):
    true_set = set(true_list)
    rec_set = set(recommended_list)
    tp = len(rec_set & true_set)
    fp = len(rec_set - true_set)
    fn = len(true_set - rec_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def average_precision(recommended_list, true_list):
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended_list):
        if item in true_list:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(true_list) if true_list else 0.0

def calculate_recall(recommended_list, true_list):
    true_set = set(true_list)
    rec_set = set(recommended_list)
    tp = len(rec_set & true_set)
    return tp / len(true_set) if true_set else 0.0

def calculate_ndcg(recommended_list, true_list, top_n):
    rel_scores = [1 if x in true_list else 0 for x in recommended_list[:top_n]]
    dcg = sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(rel_scores))
    ideal_scores = [1] * min(len(true_list), top_n)
    ideal_dcg = sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(ideal_scores))
    return dcg / ideal_dcg if ideal_dcg else 0.0

# ====================== 主流程评估类 ======================
class FMRDEvaluator:
    def __init__(self, sim_seq, ground_truth, folder_path):
        """
        sim_seq      : 已实例化的 SimSeq 对象
        ground_truth : dict, {filename: [同组的其他文件...], ...}
        folder_path  : .h5 文件所在的根目录
        """
        self.sim_seq = sim_seq
        self.ground_truth = ground_truth
        self.file_vector_map = self._precompute_file_vectors(folder_path)
        self.filenames = list(self.file_vector_map.keys())
        self.sim_matrix = self._build_similarity_matrix()

    def _precompute_file_vectors(self, folder_path):
        """遍历 folder_path 下的 .h5 文件，预计算向量，存于 file_vector_map"""
        file_vector_map = {}
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if not fname.endswith('.h5'):
                    continue
                file_path = os.path.join(root, fname)
                vec = self.sim_seq.extract_file_representation(file_path)
                if vec is not None:
                    file_vector_map[fname] = vec
        return file_vector_map

    def _build_similarity_matrix(self):
        """构建相似度矩阵 O(N^2)。"""
        sim_matrix = {}
        for target in tqdm(self.filenames, desc="Building similarity matrix"):
            target_vec = self.file_vector_map[target]
            similarities = []
            for candidate in self.filenames:
                if candidate == target:
                    continue
                cand_vec = self.file_vector_map[candidate]
                sim = self._cosine_sim(target_vec, cand_vec)
                similarities.append((candidate, sim))

            # 排序: 由大到小
            sim_matrix[target] = sorted(similarities, key=lambda x: x[1], reverse=True)
        return sim_matrix

    def _cosine_sim(self, vecA, vecB):
        """余弦相似度计算 (vecA, vecB 均为 numpy 数组)"""
        a_t = torch.tensor(vecA, dtype=torch.float32)
        b_t = torch.tensor(vecB, dtype=torch.float32)
        return F.cosine_similarity(a_t, b_t, dim=0).item()

    def evaluate(self, top_n=10):
        """根据 ground_truth 对每个文件做评估"""
        F1_list, AP_list, Recall_list, NDCG_list = [], [], [], []

        for target_file in tqdm(self.ground_truth, desc="Evaluating"):
            # 如果没在 file_vector_map 中找到对应文件，则跳过
            if target_file not in self.sim_matrix:
                continue

            # 真实同组文件（排除自身）
            true_group = [f for f in self.ground_truth[target_file] if f != target_file]
            if not true_group:
                continue

            # 推荐列表: 取相似度排序最高的 top_n (且排除自身)
            recommended = [item[0] for item in self.sim_matrix[target_file]]
            recommended = [f for f in recommended if f != target_file][:top_n]

            # 计算指标
            F1_list.append(calculate_fbeta_score(recommended, true_group))
            AP_list.append(average_precision(recommended, true_group))
            Recall_list.append(calculate_recall(recommended, true_group))
            NDCG_list.append(calculate_ndcg(recommended, true_group, top_n))

        # 计算平均
        return {
            'Mean F1': np.mean(F1_list),
            'MAP': np.mean(AP_list),
            'Mean Recall': np.mean(Recall_list),
            'Mean NDCG': np.mean(NDCG_list)
        }

def main():
    parser = argparse.ArgumentParser("Seq_FMRD Evaluation")
    parser.add_argument("--ground_truth", default=r"/home/vllm/encode/data/Seq/Seq_truth_2048.json",
                        help="Path to ground truth JSON")
    parser.add_argument("--folder_path", default=r"/home/vllm/encode/data/Seq/TEST_2048",
                        help="Folder containing .h5 files")
    parser.add_argument("--model_path", default=r"/home/vllm/encode/checkpoints/Seq/Seq_batch_size64.pth",
                        help="Trained model checkpoint path")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top similar files to retrieve")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    args = parser.parse_args()

    # 1) 初始化 SimSeq
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    sim_seq = SimSeq(args.model_path, device=device)

    # 2) 加载 ground truth
    #    假设 JSON 文件是一个列表，每个元素是一个文件组
    #    比如: [["fileA.h5", "fileB.h5"], ["fileC.h5", "fileD.h5"], ...]
    with open(args.ground_truth, 'r') as f:
        raw_truth = json.load(f)

    # 转换成 { "fileA.h5": ["fileB.h5"], "fileB.h5": ["fileA.h5"], ... } 的形式
    ground_truth = {}
    for group in raw_truth:
        for file in group:
            ground_truth[file] = [f for f in group if f != file]

    # 3) 构建评估器并计算指标
    evaluator = FMRDEvaluator(sim_seq, ground_truth, args.folder_path)
    metrics = evaluator.evaluate(top_n=args.top_n)

    # 4) 输出结果
    print("\n===== Evaluation Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
