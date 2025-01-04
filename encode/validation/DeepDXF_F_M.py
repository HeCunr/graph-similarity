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

def calculate_fbeta_score(recommended_list, true_list, beta=1):
    """
    计算 F-beta 分数
    Args:
        recommended_list: 推荐的文件列表
        true_list: ground truth 文件列表
        beta: F-beta 参数
    Returns:
        f_beta: F-beta 分数
    """
    tp = len(set(recommended_list) & set(true_list))  # True Positive
    fp = len(set(recommended_list) - set(true_list))  # False Positive
    fn = len(set(true_list) - set(recommended_list))  # False Negative

    precision = tp / (tp + fp) if tp + fp > 0 else 0  # 精度
    recall = tp / (tp + fn) if tp + fn > 0 else 0     # 召回率

    beta_squ = beta ** 2
    denominator = (beta_squ * precision) + recall
    if denominator == 0:
        return 0.0  # 如果分母为零，返回 0.0
    f_beta = (1 + beta_squ) * (precision * recall) / denominator  # F-beta 分数
    return f_beta

def average_precision(recommended_list, true_list):
    """
    计算平均精度 (Average Precision)
    Args:
        recommended_list: 推荐的文件列表
        true_list: ground truth 文件列表
    Returns:
        ap: 平均精度
    """
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended_list):
        if item in true_list:
            hits += 1
            precision = hits / (i + 1)
            precision_sum += precision
    if not true_list:
        return 0.0
    return precision_sum / len(true_list)

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

    # 计算 F1 分数和 MAP
    F1_scores = []
    APs = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name not in ground_truth:  # 如果文件不在 ground truth 中，跳过
                continue

            # 查找最相似的文件
            top_similar_files = find_top_similar_files(sim_dxf, file_path, folder_path, top_n)
            recommended_list = [os.path.basename(file_path) for file_path, _ in top_similar_files]
            true_list = ground_truth[file_name]

            # 计算 F1 分数和 AP
            f1 = calculate_fbeta_score(recommended_list, true_list, beta=1)
            ap = average_precision(recommended_list, true_list)
            F1_scores.append(f1)
            APs.append(ap)
            print(f"File: {file_name}, F1 Score: {f1:.4f}, AP: {ap:.4f}")

    # 计算平均 F1 分数和 MAP
    mean_f1 = np.mean(F1_scores)
    map_score = np.mean(APs)
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"MAP: {map_score:.4f}")

if __name__ == "__main__":
    main()