import torch
import json
import os
import numpy as np
import torch.nn.functional as F
from model.layers.DenseGraphMatching import GraphMatchNetwork
from config.GF_config import gf_args


def load_graphs_from_json(file_path):
    """
    从给定的JSON文件读取多组图的信息并解析为节点特征和邻接矩阵
    :param file_path: JSON文件路径
    :return: 节点特征矩阵列表和邻接矩阵列表
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

        for idx, succ_list in enumerate(graph_data['succs']):
            for succ in succ_list:
                adj_matrix[idx][succ] = 1

        node_features_list.append(node_features)
        adj_matrix_list.append(adj_matrix)

    return node_features_list, adj_matrix_list

def node_list(file1_path, file2_path):
    with open(file1_path) as f1, open(file2_path) as f2:
        data1 = [json.loads(line) for line in f1]
        data2 = [json.loads(line) for line in f2]

    list_1 = []
    list_2 = []

    ENTITY_TYPES = ['LINE', 'SPLINE', 'CIRCLE', 'ARC', 'ELLIPSE', 'MTEXT', 'LEADER', 'HATCH', 'DIMENSION', 'SOLID']

    # 逐行比较两个文件
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

    # 处理剩余的项
    while i < len(data1):
        list_1.append(1)
        list_2.append(0)
        i += 1

    while j < len(data2):
        list_1.append(0)
        list_2.append(1)
        j += 1

    return list_1, list_2

def compute_dxf_similarity(file1_path, file2_path):
    # 加载邻接表，特征矩阵
    graph1_features_list, graph1_adj_list = load_graphs_from_json(file1_path)
    graph2_features_list, graph2_adj_list = load_graphs_from_json(file2_path)

    # 获取节点列表
    list_1, list_2 = node_list(file1_path, file2_path)

    with torch.no_grad():
        # 初始化列表用于保存每个图的特征向量
        feature1_list = []
        feature2_list = []

        for graph1_features, graph1_adj in zip(graph1_features_list, graph1_adj_list):
            # 将它们转化为 PyTorch Tensor，并移动到设备上
            graph1_features = torch.FloatTensor(graph1_features).to(device)
            graph1_adj = torch.FloatTensor(graph1_adj).to(device)
            # 通过模型的前向传递计算每个图的特征向量
            feature = model(batch_x_p=graph1_features, batch_adj_p=graph1_adj)
            agg_feature = torch.mean(feature, dim=1)
            feature1_list.append(agg_feature)

        for graph2_features, graph2_adj in zip(graph2_features_list, graph2_adj_list):
            graph2_features = torch.FloatTensor(graph2_features).to(device)
            graph2_adj = torch.FloatTensor(graph2_adj).to(device)
            feature = model(batch_x_p=graph2_features, batch_adj_p=graph2_adj)
            agg_feature = torch.mean(feature, dim=1)
            feature2_list.append(agg_feature)

        # 特征拼接
        zero_feature = torch.zeros_like(feature1_list[0])  # 生成一个全零特征向量作为基准

        # 用于存储拼接后的特征向量
        concatenated_features_1 = []
        concatenated_features_2 = []
        for val in list_1:
            if val == 1:
                concatenated_features_1.append(feature1_list.pop(0))  # 直接拼接特征向量
            else:
                concatenated_features_1.append(zero_feature)  # 拼接全零特征向量

        for val in list_2:
            if val == 1:
                concatenated_features_2.append(feature2_list.pop(0))  # 直接拼接特征向量
            else:
                concatenated_features_2.append(zero_feature)  # 拼接全零特征向量

        # 将所有特征向量拼接在一起
        concatenated_features_1 = torch.cat(concatenated_features_1, dim=1)
        concatenated_features_2 = torch.cat(concatenated_features_2, dim=1)

        # 计算余弦相似度
        sim = F.cosine_similarity(concatenated_features_1, concatenated_features_2, dim=1).clamp(min=-1, max=1)

    return sim.item()

# 计算图对的相似度
def find_top_similar_files(target_file_path, folder_path, top_n= 20):
    similarity_scores = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            similarity = compute_dxf_similarity(target_file_path, file_path)
            similarity_scores.append((file_path, similarity))

    top_similar_files = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]

    return top_similar_files

def average_precision(recommended, actual):
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended):
        if item in actual:
            hits += 1
            precision = hits / (i + 1)
            precision_sum += precision
    if not actual:
        return 0.0
    return precision_sum / len(actual)

def calculate_fbeta_score(recommended_list, true_list, beta):
    # 计算 True Positive (TP), False Positive (FP), False Negative (FN)
    tp = len(set(recommended_list) & set(true_list))
    fp = len(set(recommended_list) - set(true_list))
    fn = len(set(true_list) - set(recommended_list))

    # 计算 Precision, Recall, 和 F1 Score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    beta_squ = beta ** 2
    f_beta = (1 + beta_squ) * (precision * recall) / ((beta_squ * precision) + recall)

    return f_beta

def find_element(json_file, target_element):
    with open(json_file, 'r') as file:
        data = json.load(file)

    for sublist in data:
        if target_element in sublist:
            return sublist

    return None


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model_path = '/home/vllm/encode/pretrained/GF/bacth_size_128/checkpoints/final_model.pt'

    model = GraphMatchNetwork(node_init_dims=6, args=gf_args).to(device)
    checkpoint = torch.load(model_path, map_location= device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    json_file = '/home/vllm/encode/cl_data/validation/ground_truth.json'
    folder_path = "/home/vllm/encode/cl_data/discreted_dataset/TEST"

    F1_score = []
    AP = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            top_similar_files = find_top_similar_files(file_path, folder_path, 20)

            pred_file = []
            for file_path, similarity_score in top_similar_files:
                file = os.path.basename(file_path)
                pred_file.append(file)

            ground_truth = find_element(json_file, file_name)
            f1 = calculate_fbeta_score(ground_truth, pred_file, 1)
            ap = average_precision(pred_file, ground_truth)
            F1_score.append(f1)
            AP.append(ap)
            print(f1)
            print(ap)

    Mean_F1_score = np.mean(F1_score)
    MAP = np.mean(AP)
    print('测试集平均F1分数为：', Mean_F1_score)
    print('测试集MAP结果为：', MAP)

