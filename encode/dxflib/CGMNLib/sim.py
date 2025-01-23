import torch
import numpy as np
import json
import os
import torch.nn.functional as F
from model.DenseGraphMatching import HierarchicalGraphMatchNetwork
from dxf_cfg_config import cfg_args

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
        concatenated_features_1 = torch.cat(concatenated_features_1, dim = 1)
        concatenated_features_2 = torch.cat(concatenated_features_2, dim = 1)

        # 计算余弦相似度
        sim = F.cosine_similarity(concatenated_features_1, concatenated_features_2, dim=1).clamp(min=-1, max=1)

    return sim.item()  # 返回相似度得分

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    BestModel_FILE ='/mnt/csip-108/cl/lf_dra_mat/CGMN/CFGLogs/Epoch_50.BestModel'

    model = HierarchicalGraphMatchNetwork(node_init_dims= 6, arguments=cfg_args, device=device).to(device)
    model.load_state_dict(torch.load(BestModel_FILE))
    model.eval()

    # 计算图对的相似度
    def find_top_similar_files(target_file_path, folder_path, top_n=10):
        similarity_scores = []

        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                similarity = compute_dxf_similarity(target_file_path, file_path)
                similarity_scores.append((file_path, similarity))

        top_similar_files = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]

        return top_similar_files


    target_file = "/mnt/csip-108/cl/lf_dra_mat/CGMN/data/te/QFN9LM(NiPdAu)-439 Rev1_5.json"
    folder = "/mnt/csip-108/cl/lf_dra_mat/CGMN/data/test"

    top_similar_files = find_top_similar_files(target_file, folder)

    print(f"检索文件：{target_file}")
    for file_path, similarity_score in top_similar_files:
        print(f"文件：{file_path}，相似度：{similarity_score}")