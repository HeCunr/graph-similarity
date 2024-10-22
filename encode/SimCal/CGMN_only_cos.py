# !/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import json

# 定义函数读取txt文件并提取图的信息
def load_graph_from_txt(file_path):
    """
    从给定的txt文件读取图的信息并解析为节点特征和邻接矩阵
    :param file_path: txt文件路径
    :return: 节点特征矩阵和邻接矩阵
    """
    with open(file_path, 'r') as f:
        graph_data = json.loads(f.read().strip())  # 读取并解析JSON格式的数据

    num_nodes = graph_data['n_num']
    node_features = np.array(graph_data['features'])  # 提取节点特征
    adj_matrix = np.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵为全0矩阵

    for idx, succ_list in enumerate(graph_data['succs']):
        for succ in succ_list:
            adj_matrix[idx][succ] = 1  # 根据succs填充邻接矩阵

    return node_features, adj_matrix

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 从两个txt文件中加载图的信息
file_path1 = '/mnt/share/CGMN/CGMN/data/CFG/OpenSSL_11ACFG_min3_max13/valid/QFN28LK(Cu)-90-450Rev1_2.txt'
# 替换为你的第一个txt文件路径
file_path2 =  '/mnt/share/CGMN/CGMN/data/CFG/OpenSSL_11ACFG_min3_max13/valid/QFN28LK(Cu)-90-450Rev1_2.txt'

# 替换为你的第二个txt文件路径
print(file_path1)
print(file_path2)
graph1_features, graph1_adj = load_graph_from_txt(file_path1)
graph2_features, graph2_adj = load_graph_from_txt(file_path2)

# 将它们转化为 PyTorch Tensor，并移动到设备上
graph1_features = torch.FloatTensor(graph1_features).to(device)
graph2_features = torch.FloatTensor(graph2_features).to(device)

import torch.nn.functional as functional

def compute_graph_similarity(graph1_features, graph2_features):
    """
    直接计算两个图的节点特征矩阵的余弦相似度
    :param graph1_features: 图1的节点特征
    :param graph2_features: 图2的节点特征
    :return: 两个图的相似度
    """
    # 计算图1和图2节点特征矩阵的平均值
    agg_p = torch.mean(graph1_features, dim=0)
    agg_h = torch.mean(graph2_features, dim=0)

    # 计算余弦相似度
    sim_score = functional.cosine_similarity(agg_p.unsqueeze(0), agg_h.unsqueeze(0), dim=1).clamp(min=-1, max=1)

    return sim_score.item()  # 返回相似度得分

# 计算图对的相似度
similarity_score = compute_graph_similarity(graph1_features, graph2_features)
print(f"图对的相似度为: {similarity_score}")
