import torch
import json
import os
import numpy as np
from SimCal.SimDeepDXF import SimDeepDXF

def load_ground_truth(ground_truth_path):
    """
    加载 ground truth 数据
    Args:
        ground_truth_path: ground truth 文件路径
    Returns:
        ground_truth: 字典，键为文件名，值为 ground truth 列表
    """
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)

    ground_truth = {}
    for group in ground_truth_data:
        for file in group:
            if file not in ground_truth:
                ground_truth[file] = group
    return ground_truth

def calculate_recall(recommended_list, true_list):
    """
    计算 Recall
    Args:
        recommended_list: 推荐的文件列表
        true_list: ground truth 文件列表
    Returns:
        recall: Recall 分数
    """
    tp = len(set(recommended_list) & set(true_list))  # True Positive
    fn = len(set(true_list) - set(recommended_list))  # False Negative
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return recall

def calculate_ndcg(recommended_list, true_list, top_n):
    """
    计算 NDCG
    Args:
        recommended_list: 推荐的文件列表
        true_list: ground truth 文件列表
        top_n: 考虑的 top N 个推荐
    Returns:
        ndcg: NDCG 分数
    """
    def dcg(relevance_scores):
        return sum((2 ** score - 1) / np.log2(idx + 2) for idx, score in enumerate(relevance_scores))

    relevance_scores = [1 if item in true_list else 0 for item in recommended_list[:top_n]]
    dcg_score = dcg(relevance_scores)
    ideal_relevance_scores = [1] * min(len(true_list), top_n)
    ideal_dcg_score = dcg(ideal_relevance_scores)
    ndcg = dcg_score / ideal_dcg_score if ideal_dcg_score > 0 else 0
    return ndcg

def find_top_similar_files(sim_dxf, target_file_path, folder_path, top_n=10):
    """
    查找与目标文件最相似的文件
    Args:
        sim_dxf: SimDeepDXF 实例
        target_file_path: 目标文件路径
        folder_path: 文件夹路径
        top_n: 返回的相似文件数量
    Returns:
        top_similar_files: 最相似的文件列表，格式为 [(文件路径, 相似度)]
    """
    similarity_scores = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path == target_file_path:  # 跳过目标文件本身
                continue
            similarity = sim_dxf.compare_h5_files(target_file_path, file_path)
            if similarity is not None:
                similarity_scores.append((file_path, similarity))
    # 按相似度排序并返回前 top_n 个文件
    top_similar_files = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return top_similar_files

def main():
    # 参数设置
    ground_truth_path = "/home/vllm/encode/data/DeepDXF/DeepDXF_truth.json"
    folder_path = r"/home/vllm/encode/data/DeepDXF/TEST_4096"
    model_path = r"/home/vllm/encode/checkpoints/best_model.pth"
    top_n = 20  # 查找最相似的文件数量

    # 加载 ground truth
    ground_truth = load_ground_truth(ground_truth_path)

    # 初始化 SimDeepDXF
    sim_dxf = SimDeepDXF(model_path)

    # 计算 Recall 和 NDCG
    recalls = []
    ndcgs = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name not in ground_truth:  # 如果文件不在 ground truth 中，跳过
                continue

            # 查找最相似的文件
            top_similar_files = find_top_similar_files(sim_dxf, file_path, folder_path, top_n)
            recommended_list = [os.path.basename(file_path) for file_path, _ in top_similar_files]
            true_list = ground_truth[file_name]

            # 计算 Recall 和 NDCG
            recall = calculate_recall(recommended_list, true_list)
            ndcg = calculate_ndcg(recommended_list, true_list, top_n)
            recalls.append(recall)
            ndcgs.append(ndcg)
            print(f"File: {file_name}, Recall@{top_n}: {recall:.4f}, NDCG@{top_n}: {ndcg:.4f}")

    # 计算平均 Recall 和 NDCG
    mean_recall = np.mean(recalls)
    mean_ndcg = np.mean(ndcgs)
    print(f"Mean Recall@{top_n}: {mean_recall:.4f}")
    print(f"Mean NDCG@{top_n}: {mean_ndcg:.4f}")

if __name__ == "__main__":
    main()