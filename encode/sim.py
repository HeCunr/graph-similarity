# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import torch
import h5py
import numpy as np
import torch.nn.functional as F
from model.layers.DenseGraphMatching import HierarchicalGraphMatchNetwork
from model.transformer_encoder import DXFTransformer
from config.CGMN_cfg_config import cfg_args
from model.CGMN_dataset import CFGDataset
from utils.CGMN_utils import graph

class SimilarityCalculator:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # 加载保存的模型
        self.state_dict = torch.load(model_path, map_location=self.device)

        # 初始化CGMN模型
        self.cgmn_model = HierarchicalGraphMatchNetwork(
            node_init_dims=cfg_args.graph_init_dim,
            arguments=cfg_args,
            device=self.device
        ).to(self.device)
        self.cgmn_model.load_state_dict(self.state_dict['cgmn_model'])
        self.cgmn_model.eval()

        # 初始化DeepDXF模型
        self.dxf_model = DXFTransformer().to(self.device)
        self.dxf_model.load_state_dict(self.state_dict['dxf_model'])
        self.dxf_model.eval()

        # 设置权重
        self.cgmn_weight = 0.5
        self.dxf_weight = 0.5

    def load_json_to_graph(self, json_path):
        """将特殊格式的json文件加载为图结构"""
        with open(json_path, 'r') as f:
            # 读取整个文件内容
            content = f.read().strip()
            # 将字符串转换为Python字典
            data = eval(content)  # 使用eval而不是json.loads因为格式特殊

        # 获取节点数量和邻接关系
        n_num = data['n_num']
        succs = data['succs']
        features = data['features']

        # 创建图对象
        g = graph(node_num=n_num, label=0)

        # 添加节点特征和边
        for i in range(n_num):
            g.features[i] = np.array(features[i])  # 使用文件中的特征
            for succ in succs[i]:  # 添加边
                g.add_edge(i, succ)

        return g

    def load_h5_file(self, h5_path):
        """加载H5文件"""
        with h5py.File(h5_path, 'r') as f:
            dxf_vec = f['dxf_vec'][:]  # [1, 512, 39]
            # 分离entity_type和entity_params
            entity_type = dxf_vec[:, :, 0]  # [1, 512]
            entity_params = dxf_vec[:, :, 1:]  # [1, 512, 38]
            return (torch.tensor(entity_type, dtype=torch.long).to(self.device),
                    torch.tensor(entity_params, dtype=torch.float32).to(self.device))

    def calculate_cgmn_similarity(self, json_path1, json_path2):
        """计算CGMN模型的相似度"""
        with torch.no_grad():
            # 加载图结构
            graph1 = self.load_json_to_graph(json_path1)
            graph2 = self.load_json_to_graph(json_path2)

            # 准备输入数据
            x1 = torch.FloatTensor(graph1.features).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(graph2.features).unsqueeze(0).to(self.device)

            adj1 = torch.zeros((1, len(graph1.features), len(graph1.features))).to(self.device)
            adj2 = torch.zeros((1, len(graph2.features), len(graph2.features))).to(self.device)

            # 填充邻接矩阵
            for i in range(len(graph1.features)):
                for j in graph1.succs[i]:
                    adj1[0, i, j] = 1
            for i in range(len(graph2.features)):
                for j in graph2.succs[i]:
                    adj2[0, i, j] = 1

            # 获取特征表示
            feature1 = self.cgmn_model(batch_x_p=x1, batch_adj_p=adj1)
            feature2 = self.cgmn_model(batch_x_p=x2, batch_adj_p=adj2)

            # 计算相似度
            agg1 = torch.mean(feature1, dim=1)
            agg2 = torch.mean(feature2, dim=1)
            similarity = F.cosine_similarity(agg1, agg2, dim=1).item()

            return similarity

    def calculate_dxf_similarity(self, h5_path1, h5_path2):
        """计算DeepDXF模型的相似度"""
        with torch.no_grad():
            # 加载H5文件
            entity_type1, entity_params1 = self.load_h5_file(h5_path1)
            entity_type2, entity_params2 = self.load_h5_file(h5_path2)

            # 打印输入形状
            print(f"entity_type1 shape: {entity_type1.shape}")
            print(f"entity_params1 shape: {entity_params1.shape}")

            # 获取特征表示
            z1, _, _ = self.dxf_model(entity_type1, entity_params1)
            z2, _, _ = self.dxf_model(entity_type2, entity_params2)

            # 打印特征向量
            print(f"z1 shape: {z1.shape}")
            print(f"z2 shape: {z2.shape}")

            # 计算相似度
            similarity = F.cosine_similarity(z1, z2, dim=1).item()
            print(f"Raw cosine similarity: {similarity}")

            # 映射到[0,1]范围
            similarity = (similarity + 1) / 2
            print(f"Normalized similarity: {similarity}")

            return similarity

    def calculate_similarity(self, json_path1, h5_path1, json_path2, h5_path2):
        """计算总的相似度分数"""
        # 计算CGMN相似度
        cgmn_similarity = self.calculate_cgmn_similarity(json_path1, json_path2)

        # 计算DeepDXF相似度
        dxf_similarity = self.calculate_dxf_similarity(h5_path1, h5_path2)

        # 计算加权总分
        total_similarity = (self.cgmn_weight * cgmn_similarity +
                            self.dxf_weight * dxf_similarity)

        return {
            'total_similarity': total_similarity,
            'cgmn_similarity': cgmn_similarity,
            'dxf_similarity': dxf_similarity
        }

def main():
    # 参数设置
    model_path = './checkpoints/combined_model.pth'  # 训练好的模型路径

    # 测试文件路径
    json_path1 = r'/mnt/share/DeepDXF_CGMN/encode/data/test/437_1_3.json'
    h5_path1 = r'/mnt/share/DeepDXF_CGMN/encode/data/test/437_1_3.h5'
    json_path2 = r'/mnt/share/DeepDXF_CGMN/encode/data/test/517_1_3.json'
    h5_path2 = r'/mnt/share/DeepDXF_CGMN/encode/data/test/517_1_3.h5'

    # 创建相似度计算器
    calculator = SimilarityCalculator(model_path)

    # 计算相似度
    similarity_scores = calculator.calculate_similarity(
        json_path1, h5_path1,
        json_path2, h5_path2
    )

    # 打印结果
    print("\nSimilarity Scores:")
    print(json_path1, h5_path1)
    print(json_path2, h5_path2)
    print(f"Total Similarity: {similarity_scores['total_similarity']:.4f}")
    print(f"CGMN Similarity: {similarity_scores['cgmn_similarity']:.4f}")
    print(f"DeepDXF Similarity: {similarity_scores['dxf_similarity']:.4f}")

if __name__ == '__main__':
    main()