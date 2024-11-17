#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import json
import argparse
from model.layers.DenseGraphMatching import HierarchicalGraphMatchNetwork
from config.GF_config import get_args
from utils.GF_utils import graph

def load_graph_from_json(file_path):
    """
    从给定的文件读取图的信息并解析为节点特征和邻接矩阵
    :param file_path: 文件路径
    :return: 图对象
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 只读取第一行
            if lines:
                graph_data = json.loads(lines[0].strip())
            else:
                raise ValueError("Empty file")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        raise
    except Exception as e:
        print(f"Error reading file: {e}")
        raise

    # 创建图对象，使用src作为图的名称
    current_graph = graph(node_num=graph_data['n_num'], name=graph_data.get('src', ''))

    # 设置节点特征
    for u in range(graph_data['n_num']):
        current_graph.features[u] = np.array(graph_data['features'][u])
        # 添加边
        for v in graph_data['succs'][u]:
            current_graph.add_edge(u, v)

    return current_graph

def prepare_graph_data(graph_obj):
    """
    将图对象转换为模型所需的格式
    :param graph_obj: 图对象
    :return: 特征矩阵和邻接矩阵
    """
    # 准备特征矩阵
    features = np.array([feat for feat in graph_obj.features])
    features = np.expand_dims(features, axis=0)  # 添加batch维度

    # 准备邻接矩阵
    n = len(graph_obj.features)
    adj_matrix = np.zeros((n, n))
    for u in range(n):
        for v in graph_obj.succs[u]:
            adj_matrix[u][v] = 1
    adj_matrix = np.expand_dims(adj_matrix, axis=0)  # 添加batch维度

    return features, adj_matrix

def normalize_similarity(similarity):
    """
    将相似性分数归一化到[0,1]范围
    :param similarity: 原始相似性分数
    :return: 归一化后的相似性分数
    """
    # 使用min-max归一化
    return (similarity + 1) / 2

def compute_graph_similarity(graph1_path, graph2_path, model_path, device='cuda'):
    """
    计算两个图的相似度
    :param graph1_path: 第一个图的文件路径
    :param graph2_path: 第二个图的文件路径
    :param model_path: 模型文件路径
    :param device: 计算设备
    :return: 相似度得分 (范围[0,1])
    """
    # 加载图
    try:
        graph1 = load_graph_from_json(graph1_path)
        graph2 = load_graph_from_json(graph2_path)
    except Exception as e:
        print(f"Error loading graphs: {e}")
        return 0.0  # 错误情况下返回最小相似度

    # 准备数据
    features1, adj1 = prepare_graph_data(graph1)
    features2, adj2 = prepare_graph_data(graph2)

    # 转换为tensor并移动到设备
    features1 = torch.FloatTensor(features1).to(device)
    adj1 = torch.FloatTensor(adj1).to(device)
    features2 = torch.FloatTensor(features2).to(device)
    adj2 = torch.FloatTensor(adj2).to(device)

    # 检查CUDA是否可用
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        device = 'cpu'

    # 初始化模型
    node_init_dims = features1.shape[-1]  # 获取节点特征维度
    args = get_args()  # 获取配置参数
    model = HierarchicalGraphMatchNetwork(
        node_init_dims=node_init_dims,
        arguments=args,
        device=device
    ).to(device)

    try:
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 计算图的嵌入
        with torch.no_grad():
            # 通过模型获取图的表示
            feature_p = model(batch_x_p=features1, batch_adj_p=adj1)
            feature_h = model(batch_x_p=features2, batch_adj_p=adj2)

            # 计算余弦相似度
            agg_p = torch.mean(feature_p, dim=1)  # 平均池化得到图级别的表示
            agg_h = torch.mean(feature_h, dim=1)

            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(agg_p, agg_h, dim=1)
            # 将相似度限制在[-1, 1]范围内
            similarity = torch.clamp(similarity, min=-1, max=1)
            # 归一化到[0, 1]范围
            normalized_similarity = normalize_similarity(similarity)

    except Exception as e:
        print(f"Error during computation: {e}")
        return 0.0  # 错误情况下返回最小相似度

    return normalized_similarity.item()

def main():
    # 使用 get_args() 获取所有配置参数
    args = get_args()

    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'

    try:
        # 计算相似度
        similarity = compute_graph_similarity(
            args.graph1,
            args.graph2,
            args.model,
            args.device
        )
        print(f"\nSimilarity score between the graphs: {similarity:.4f}")
    except Exception as e:
        print(f"Error in computation: {e}")
        return 1

    return 0

if __name__ == '__main__':
    main()