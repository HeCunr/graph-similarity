#dataset/Geom_dataset.py

import copy
import json
import os
from typing import List, Tuple
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset
import random

class GraphData:
    def __init__(self, node_num: int, name: str = None):
        self.node_num = node_num
        self.name = name
        self.features = np.zeros((node_num, 0))
        self.adj = nx.Graph()

        # 让networkx里真的存在 [0..node_num-1] 这些节点
        self.adj.add_nodes_from(range(node_num))

        self.matrices = None

    def add_edge(self, u: int, v: int):
        self.adj.add_edge(u, v)


class GeomDataset:
    def __init__(self, data_dir: str, args):
        """
        Initialize Geom dataset

        Args:
            data_dir: Directory containing the graph data
            args: Arguments from Geom_config
        """
        self.args = args
        self.data_dir = data_dir

        # 1. 读取所有图
        self.graphs = self.load_raw_graphs()
        self.num_graphs = len(self.graphs)

        # 2. 求所有图中最大节点数
        self.max_nodes = self._get_max_nodes()

        # 3. 处理图（补齐特征矩阵、邻接矩阵、mask 等）
        self.graphs = self.process_loaded_graphs()

        # 4. 按照 7:1.5:1.5 拆分数据
        # 先随机打乱索引
        indices = np.arange(self.num_graphs)
        np.random.seed(args.seed)
        np.random.shuffle(indices)

        train_end = int(self.num_graphs * args.train_split)  # 0.7
        val_end = int(self.num_graphs * (args.train_split + args.val_split))  # 0.7 + 0.15 = 0.85

        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

        # 也可直接存储相应的图对象以便后续获取
        self.train_graphs = [self.graphs[i] for i in self.train_indices]
        self.val_graphs = [self.graphs[i] for i in self.val_indices]
        self.test_graphs = [self.graphs[i] for i in self.test_indices]

    def load_raw_graphs(self) -> List[GraphData]:
        """Load graphs without processing"""
        graphs = []
        for file in os.listdir(self.data_dir):
            if not file.endswith('.json'):
                continue
            with open(os.path.join(self.data_dir, file)) as f:
                for line in f:
                    g_info = json.loads(line.strip())
                    n_num = g_info['n_num']

                    # ---- 如果节点数是 0，直接跳过 ----
                    if n_num <= 0:
                        print(f"[Warning] Skipping graph {g_info.get('src')} because n_num={n_num} <= 0")
                        continue

                    graph = GraphData(node_num=n_num, name=g_info['src'])

                    if 'features' not in g_info:
                        print(f"[Warning] Skipping graph {g_info.get('src')} (missing features)")
                        continue
                    graph.features = np.array(g_info['features'])

                    # 添加边
                    has_any_edge = False
                    for u in range(n_num):
                        if u >= len(g_info['succs']):
                            continue
                        for v in g_info['succs'][u]:
                            graph.add_edge(u, v)
                            has_any_edge = True

                    #跳过完全没有边的图，可加判断
                    if not has_any_edge:
                        print(f"[Warning] Skipping graph {g_info.get('src')} because it has no edges.")
                        continue

                    graphs.append(graph)
        return graphs


    def process_loaded_graphs(self) -> List[GraphData]:
        """Process all loaded graphs"""
        processed_graphs = []
        for i, graph in enumerate(self.graphs):
            feature_matrix, adj_matrix, mask = self._process_single_graph(graph)
            graph_copy = copy.deepcopy(graph)
            graph_copy.matrices = (feature_matrix, adj_matrix, mask)
            processed_graphs.append(graph_copy)
        return processed_graphs

    def _get_max_nodes(self) -> int:
        """Get maximum number of nodes across all graphs"""
        return max(g.node_num for g in self.graphs)

    def _process_single_graph(self, graph: GraphData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对单个图进行处理：对特征矩阵和邻接矩阵按最大节点数量进行对齐，并生成掩码。
        """

        # 1) 从 networkx.Graph 构建邻接矩阵
        adj = nx.adjacency_matrix(graph.adj).toarray()

        # 2) 加 self-loop
        np.fill_diagonal(adj, 1)

        actual_n = adj.shape[0]  # 实际节点数
        max_n = self.args.graph_size_max  # 允许的最大节点数
        feature_dim = self.args.graph_init_dim  # 每个节点的特征维度

        # 如果实际节点数大于 max_n，需要截断 (也可选择抛异常)
        if actual_n > max_n:
            print(f"[Warning] Graph {graph.name} has {actual_n} nodes, exceed max {max_n}, slicing to {max_n}")
            # 截断邻接矩阵
            adj = adj[:max_n, :max_n]
            actual_n = max_n
            # 如果特征也比 max_n 大，需同步截断
            if graph.features.shape[0] > max_n:
                graph.features = graph.features[:max_n, :]

        # 3) 构建补零后的邻接矩阵（多余部分填 0）
        adj_padded = np.zeros((max_n, max_n), dtype=np.float32)
        adj_padded[:actual_n, :actual_n] = adj[:actual_n, :actual_n]

        # 4) 构建特征矩阵。用 -1 表示“没有该参数”，其余位置填入原有特征
        feature_matrix = np.full((max_n, feature_dim), -1, dtype=np.float32)
        if len(graph.features.shape) == 1:
            # 如果原始是一维，需要 reshape
            graph.features = graph.features.reshape(1, -1)

        # 同样地，如果原始特征行数 < actual_n，这里会自动把足够的行数拷贝；如果超出，也要截断
        feature_matrix[:actual_n, :] = graph.features[:actual_n, :]

        # 5) 掩码：前 actual_n 个位置为 1，其余为 0
        mask = np.zeros((max_n,), dtype=np.float32)
        mask[:actual_n] = 1.0

        return feature_matrix, adj_padded, mask



    # 以下根据需要封装一些简单的 get 函数
    def get_train_data(self):
        return self.train_graphs

    def get_val_data(self):
        return self.val_graphs

    def get_test_data(self):
        return self.test_graphs

