import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from config.GF_config import gf_args
from model.layers.DenseGraphMatching import GraphMatchNetwork

#############################################
# 1. 基础函数 - 读取 JSON 并构建图
#############################################
def load_graphs_from_json(file_path):
    """
    从给定的JSON文件读取多组图的信息并解析为节点特征和邻接矩阵
    返回:
      node_features_list: [ (n_i, feat_dim), ... ]  每个子图的节点特征
      adj_matrix_list:    [ (n_i, n_i),       ... ]  每个子图的邻接矩阵
    """
    node_features_list = []
    adj_matrix_list = []

    with open(file_path, 'r') as f:
        json_data = f.readlines()

    for line in json_data:
        graph_data = json.loads(line.strip())

        num_nodes = graph_data['n_num']
        node_features = np.array(graph_data['features'])
        adj_matrix = np.zeros((num_nodes, num_nodes))

        # 构建邻接矩阵
        for idx, succ_list in enumerate(graph_data['succs']):
            for succ in succ_list:
                adj_matrix[idx][succ] = 1

        node_features_list.append(node_features)
        adj_matrix_list.append(adj_matrix)

    return node_features_list, adj_matrix_list


def node_list(file1_path, file2_path):
    """
    用于比较两个JSON文件中的"src"字段列表，返回 list_1, list_2 两个序列，
    其中1表示该位置存在对应元素，0表示填充。
    """
    with open(file1_path) as f1, open(file2_path) as f2:
        data1 = [json.loads(line) for line in f1]
        data2 = [json.loads(line) for line in f2]

    list_1 = []
    list_2 = []

    ENTITY_TYPES = ['LINE', 'SPLINE', 'CIRCLE', 'ARC', 'ELLIPSE', 'MTEXT',
                    'LEADER', 'HATCH', 'DIMENSION', 'SOLID']

    i = 0
    j = 0
    while i < len(data1) and j < len(data2):
        item1 = data1[i]
        item2 = data2[j]

        if item1['src'] == item2['src']:
            list_1.append(1)
            list_2.append(1)
            i += 1
            j += 1
        else:
            priority1 = ENTITY_TYPES.index(item1['src']) if item1['src'] in ENTITY_TYPES else len(ENTITY_TYPES)
            priority2 = ENTITY_TYPES.index(item2['src']) if item2['src'] in ENTITY_TYPES else len(ENTITY_TYPES)
            if priority1 > priority2:
                list_1.append(0)
                list_2.append(1)
                j += 1
            else:
                list_1.append(1)
                list_2.append(0)
                i += 1

    # 若有剩余
    while i < len(data1):
        list_1.append(1)
        list_2.append(0)
        i += 1
    while j < len(data2):
        list_1.append(0)
        list_2.append(1)
        j += 1

    return list_1, list_2


#############################################
# 2. 计算两个DXF文件的相似度
#############################################
def compute_dxf_similarity(file1_path, file2_path, model, device):
    """
    传入两个json文件（已按discrete生成）, 计算它们的相似度 (余弦相似度)
    """
    # 1) 加载图
    graph1_features_list, graph1_adj_list = load_graphs_from_json(file1_path)
    graph2_features_list, graph2_adj_list = load_graphs_from_json(file2_path)

    # 2) 获取节点列表-对齐
    list_1, list_2 = node_list(file1_path, file2_path)

    # 3) 前向传播，得到每个子图的聚合向量
    with torch.no_grad():
        feature1_list = []
        feature2_list = []

        for graph1_features, graph1_adj in zip(graph1_features_list, graph1_adj_list):
            graph1_features = torch.FloatTensor(graph1_features).unsqueeze(0).to(device)  # [1, N, F]
            graph1_adj = torch.FloatTensor(graph1_adj).unsqueeze(0).to(device)            # [1, N, N]
            out = model(graph1_features, graph1_adj)  # [1, N, d]
            agg_feature = torch.mean(out, dim=1)                            # [1, d]
            feature1_list.append(agg_feature)

        for graph2_features, graph2_adj in zip(graph2_features_list, graph2_adj_list):
            graph2_features = torch.FloatTensor(graph2_features).unsqueeze(0).to(device)
            graph2_adj = torch.FloatTensor(graph2_adj).unsqueeze(0).to(device)
            out = model(graph2_features, graph2_adj)
            agg_feature = torch.mean(out, dim=1)
            feature2_list.append(agg_feature)

        # 4) 拼接(对齐)特征
        zero_feature = torch.zeros_like(feature1_list[0])  # [1, d]

        concatenated_features_1 = []
        concatenated_features_2 = []

        for val in list_1:
            if val == 1:
                concatenated_features_1.append(feature1_list.pop(0))  # [1, d]
            else:
                concatenated_features_1.append(zero_feature)

        for val in list_2:
            if val == 1:
                concatenated_features_2.append(feature2_list.pop(0))
            else:
                concatenated_features_2.append(zero_feature)

        # 将列表里的 [1, d] 沿 dim=1 拼接
        concatenated_features_1 = torch.cat(concatenated_features_1, dim=1)  # [1, d * k1]
        concatenated_features_2 = torch.cat(concatenated_features_2, dim=1)  # [1, d * k2]

        # 5) 计算余弦相似度
        sim = F.cosine_similarity(concatenated_features_1, concatenated_features_2, dim=1).clamp(-1, 1)
        return sim.item()


#############################################
# 3. 找到某文件在 ground_truth.json 中所在的子列表
#############################################
def find_element(json_file, target_element):
    """
    ground_truth.json 的格式类似:
    [
      ["fileA.json", "fileB.json", "fileC.json"],
      ["fileX.json", "fileY.json"],
      ...
    ]
    若 target_element 在某个子列表内，就返回那个子列表；否则返回 None
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    for sublist in data:
        if target_element in sublist:
            return sublist
    return None


#############################################
# 4. 推荐接口 & 评估指标
#############################################
def find_top_similar_files(target_file_path, folder_path, model, device, top_n=10):
    """
    给定目标文件 target_file_path, 在 folder_path 中找最相似的 top_n 个文件
    """
    similarity_scores = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            file_path = os.path.join(root, fname)

            # 跳过自己
            if file_path == target_file_path:
                continue

            similarity = compute_dxf_similarity(target_file_path, file_path, model, device)
            similarity_scores.append((file_path, similarity))

    top_similar_files = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return top_similar_files


def average_precision(recommended, actual):
    """
    计算 AP (Average Precision)
      recommended: 推荐的文件名列表
      actual: ground_truth 列表
    """
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended):
        if item in actual:
            hits += 1
            precision = hits / (i + 1)
            precision_sum += precision
    return precision_sum / len(actual) if actual else 0.0


def calculate_fbeta_score(recommended_list, true_list, beta=1.0):
    """
    计算 Fbeta 分数 (默认 F1)
    """
    tp = len(set(recommended_list) & set(true_list))
    fp = len(set(recommended_list) - set(true_list))
    fn = len(set(true_list) - set(recommended_list))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    beta_squ = beta ** 2
    if precision == 0 and recall == 0:
        return 0.0
    f_beta = (1 + beta_squ) * (precision * recall) / ((beta_squ * precision) + recall)
    return f_beta


def calculate_recall(recommended_list, true_list):
    """
    计算 Recall
    """
    tp = len(set(recommended_list) & set(true_list))  # True Positive
    fn = len(set(true_list) - set(recommended_list))  # False Negative
    return tp / (tp + fn) if tp + fn > 0 else 0


def calculate_ndcg(recommended_list, true_list, top_n):
    """
    计算 NDCG
    """
    def dcg(relevance_scores):
        return sum((2 ** score - 1) / np.log2(idx + 2) for idx, score in enumerate(relevance_scores))

    # relevance_scores: 推荐列表在前 top_n 个位置是否在true_list中
    relevance_scores = [1 if item in true_list else 0 for item in recommended_list[:top_n]]
    dcg_score = dcg(relevance_scores)

    # 理想情况下的分数
    ideal_relevance_scores = [1] * min(len(true_list), top_n)
    ideal_dcg_score = dcg(ideal_relevance_scores)

    return dcg_score / ideal_dcg_score if ideal_dcg_score > 0 else 0


#############################################
# 5. 主流程: 加载模型 & 计算各种指标
#############################################
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载训练好的模型
    model_path = r"/home/vllm/encode/pretrained/GF/best_checkpoint.pt"
    model = GraphMatchNetwork(node_init_dims=6, args=gf_args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2) 数据所在位置
    folder_path = r"/home/vllm/encode/data/GF/discreted_dataset/TEST"
    ground_truth_json = r"/home/vllm/encode/data/GF/validation/ground_truth.json"
    top_n = 10  # 同时用于找相似文件

    # 存放各项指标
    f1_scores = []
    ap_scores = []
    recall_scores = []
    ndcg_scores = []

    # 3) 遍历测试集文件
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            target_file_path = os.path.join(root, file_name)

            # 在 ground_truth.json 中找到这个 file_name 对应的列表
            true_list = find_element(ground_truth_json, file_name)
            if not true_list:
                # 如果没有出现在 ground_truth.json 中，就跳过或视为无真值
                continue

            # 4) 获取最相似的 top_n 文件
            top_similar = find_top_similar_files(target_file_path, folder_path, model, device, top_n)
            # top_similar: [ (file_path, score), ...]
            pred_file_list = [os.path.basename(fpath) for (fpath, score) in top_similar]

            # 5) 计算各种度量: F1, AP, Recall, NDCG
            f1 = calculate_fbeta_score(pred_file_list, true_list, beta=1.0)
            ap = average_precision(pred_file_list, true_list)
            r  = calculate_recall(pred_file_list, true_list)
            nd = calculate_ndcg(pred_file_list, true_list, top_n)

            f1_scores.append(f1)
            ap_scores.append(ap)
            recall_scores.append(r)
            ndcg_scores.append(nd)

            print(f"File: {file_name}\n"
                  f"   F1   = {f1:.4f}\n"
                  f"   AP   = {ap:.4f}\n"
                  f"   Recall = {r:.4f}\n"
                  f"   NDCG@{top_n} = {nd:.4f}\n")

    # 6) 汇总平均结果
    mean_f1    = np.mean(f1_scores)    if f1_scores    else 0
    mean_ap    = np.mean(ap_scores)    if ap_scores    else 0
    mean_recall= np.mean(recall_scores)if recall_scores else 0
    mean_ndcg  = np.mean(ndcg_scores)  if ndcg_scores  else 0

    print("\n================= Summary =================")
    print(f"Mean F1 score:       {mean_f1:.4f}")
    print(f"Mean AP:             {mean_ap:.4f}")
    print(f"Mean Recall@{top_n}: {mean_recall:.4f}")
    print(f"Mean NDCG@{top_n}:   {mean_ndcg:.4f}")
