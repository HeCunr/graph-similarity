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
file_path1 = '/mnt/share/CGMN/CGMN/CFGLogs/同一类之间/2_2.json'  # 替换为你的第一个txt文件路径
file_path2 = '/mnt/share/CGMN/CGMN/CFGLogs/同一类之间/2_1.json'  # 替换为你的第二个txt文件路径
print(file_path1)
print(file_path2)
graph1_features, graph1_adj = load_graph_from_txt(file_path1)
graph2_features, graph2_adj = load_graph_from_txt(file_path2)

# 将它们转化为 PyTorch Tensor，并移动到设备上
graph1_features = torch.FloatTensor(graph1_features).to(device)
graph1_adj = torch.FloatTensor(graph1_adj).to(device)
graph2_features = torch.FloatTensor(graph2_features).to(device)
graph2_adj = torch.FloatTensor(graph2_adj).to(device)

# 加载你的模型（模型路径需替换为你自己的BestModel_FILE路径）
from model.layers.DenseGraphMatching import HierarchicalGraphMatchNetwork
from config.CGMN_cfg_config import cfg_args  # 导入配置参数

args = cfg_args  # 使用之前的模型参数
node_init_dims = graph1_features.shape[1]  # 从节点特征维度动态获取
BestModel_FILE ='/mnt/share/CGMN/CGMN/CFGLogs/OpenSSL_Min10_Max10_InitDims11_Task_classification\BestModels_Repeat_1\OpenSSL_Min10_Max10_InitDims11_Task_classification_Filter_100_100_100_Match_concat_P_100_Agg_lstm_Hidden_100_Epoch_50_Batch_5_lr_0.0001_Dropout_0.1_Global_0_with_agg_max_pool.BestModel'
# 加载模型
model = HierarchicalGraphMatchNetwork(node_init_dims=node_init_dims, arguments=args, device=device).to(device)
model.load_state_dict(torch.load(BestModel_FILE))
model.eval()  # 设置为评估模式

import torch.nn.functional as functional

def compute_graph_similarity(graph1_features, graph1_adj, graph2_features, graph2_adj):
    """
    使用训练好的模型计算两个图的相似性
    :param graph1_features: 图1的节点特征
    :param graph1_adj: 图1的邻接矩阵
    :param graph2_features: 图2的节点特征
    :param graph2_adj: 图2的邻接矩阵
    :return: 两个图的相似度
    """
    with torch.no_grad():
        # 通过模型的前向传递计算每个图的特征向量
        feature_p = model(batch_x_p=graph1_features, batch_adj_p=graph1_adj)
        feature_h = model(batch_x_p=graph2_features, batch_adj_p=graph2_adj)

        # 计算两个图的余弦相似度
        agg_h = torch.mean(feature_h, dim=1)
        agg_p = torch.mean(feature_p, dim=1)
        sim_score = functional.cosine_similarity(agg_p, agg_h, dim=1).clamp(min=-1, max=1)

    return sim_score.item()  # 返回相似度得分

# 计算图对的相似度
similarity_score = compute_graph_similarity(graph1_features, graph1_adj, graph2_features, graph2_adj)
print(f"图对的相似度为: {similarity_score}")
