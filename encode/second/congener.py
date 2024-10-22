import numpy as np
import torch
from adj_row_col import get_normalized_feature_matrix

def create_augmented_feature_matrices(feature_matrix, num_variants=6, dropout_prob=0.1):
    # 提取除第一列外的特征矩阵
    first_col = feature_matrix[:, 0].reshape(-1, 1)
    temp = feature_matrix[:, 1:]

    # 初始化列表存储变体特征矩阵和邻接表
    augmented_features = []
    adjacency_lists = []

    for _ in range(num_variants):
        # 生成遮盖后的特征矩阵副本
        temp_variant = temp.copy()

        # 使用dropout进行随机遮盖
        mask = np.random.rand(*temp_variant.shape) >= dropout_prob
        temp_variant = temp_variant * mask

        # 把遮盖后的特征矩阵大于0的元素视为有边，生成邻接表
        adj_list = {}
        for i in range(temp_variant.shape[0]):
            adj_list[i] = [j for j in range(temp_variant.shape[1]) if temp_variant[i, j] > 0]

        # 将变体特征矩阵与第一列拼接
        augmented_feature = np.hstack((first_col, temp_variant))

        # 将结果添加到列表中
        augmented_features.append(augmented_feature)
        adjacency_lists.append(adj_list)

    return augmented_features, adjacency_lists

if __name__ == '__main__':
    # 加载特征矩阵
    dxf_file_path = r'C:\Users\15653\dwg-cx\dataset\modified\DFN6LCG(NiPdAu)（321）-517  Rev1_1.dxf'
    feature_matrix = get_normalized_feature_matrix(dxf_file_path)
    feature_matrix = np.array(feature_matrix)  # 转化为 numpy 数组

    # 生成增强的特征矩阵和邻接表
    augmented_features, adjacency_lists = create_augmented_feature_matrices(feature_matrix)

    # 输出结果
    for i, (feature, adj_list) in enumerate(zip(augmented_features, adjacency_lists), 1):
        print(f"f{i} (特征矩阵):")
        print(feature)
        print(f"adj{i} (邻接表):")
        print(adj_list)
