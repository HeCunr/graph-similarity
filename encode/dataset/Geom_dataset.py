# dataset/Geom_dataset.py

import copy
import json
import os
from typing import List, Tuple
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset

class GraphData:
    def __init__(self, node_num: int, name: str = None):
        self.node_num = node_num
        self.name = name
        self.features = np.zeros((node_num, 0))
        self.adj = nx.Graph()
        self.pos2d = None

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

        # 3. 处理图（补齐特征矩阵、邻接矩阵、mask、pos2d 等）
        self.graphs = self.process_loaded_graphs()

        # 4. 按照 7:1.5:1.5 拆分数据
        indices = np.arange(self.num_graphs)
        np.random.seed(args.seed)
        np.random.shuffle(indices)

        train_end = int(self.num_graphs * args.train_split)  # 0.7
        val_end = int(self.num_graphs * (args.train_split + args.val_split))  # 0.85

        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

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

                    if n_num <= 0:
                        continue

                    graph = GraphData(node_num=n_num, name=g_info['src'])

                    # 加载特征
                    if 'features' not in g_info:
                        continue
                    graph.features = np.array(g_info['features'])

                    # 加载2D-index（若存在）
                    if '2D-index' in g_info:
                        graph.pos2d = np.array(g_info['2D-index'])  # [n_num, 2]

                    # 添加边
                    has_any_edge = False
                    for u in range(n_num):
                        if u >= len(g_info['succs']):
                            continue
                        for v in g_info['succs'][u]:
                            graph.add_edge(u, v)
                            has_any_edge = True

                    if not has_any_edge:
                        continue

                    graphs.append(graph)
        return graphs

    def _get_max_nodes(self) -> int:
        """Get maximum number of nodes across all graphs"""
        return max(g.node_num for g in self.graphs)

    def process_loaded_graphs(self) -> List[GraphData]:
        """Process all loaded graphs"""
        processed_graphs = []
        for i, graph in enumerate(self.graphs):
            feature_matrix, adj_matrix, mask, pos2d_matrix = self._process_single_graph(graph)
            graph_copy = copy.deepcopy(graph)
            graph_copy.matrices = (feature_matrix, adj_matrix, mask, pos2d_matrix)
            processed_graphs.append(graph_copy)
        return processed_graphs

    def _process_single_graph(self, graph: GraphData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对单个图进行处理：对特征矩阵/邻接矩阵按最大节点数量进行对齐，并生成掩码。
        同时对 pos2d 做同样的对齐/截断。
        """
        adj = nx.adjacency_matrix(graph.adj).toarray()
        np.fill_diagonal(adj, 1)  # 加 self-loop

        actual_n = adj.shape[0]
        max_n = self.args.graph_size_max
        feature_dim = self.args.graph_init_dim

        if actual_n > max_n:
            # 截断邻接矩阵
            adj = adj[:max_n, :max_n]
            actual_n = max_n
            # 同步截断特征
            if graph.features.shape[0] > max_n:
                graph.features = graph.features[:max_n, :]

            # 同步截断pos2d
            if graph.pos2d is not None and graph.pos2d.shape[0] > max_n:
                graph.pos2d = graph.pos2d[:max_n, :]

        # 邻接矩阵补零
        adj_padded = np.zeros((max_n, max_n), dtype=np.float32)
        adj_padded[:actual_n, :actual_n] = adj[:actual_n, :actual_n]

        # 特征矩阵补 -1
        feature_matrix = np.full((max_n, feature_dim), -1, dtype=np.float32)
        if len(graph.features.shape) == 1:
            graph.features = graph.features.reshape(1, -1)
        feature_matrix[:actual_n, :] = graph.features[:actual_n, :]

        # 掩码
        mask = np.zeros((max_n,), dtype=np.float32)
        mask[:actual_n] = 1.0

        # pos2d也进行补 -1（若没有，则默认全 -1）
        pos2d_matrix = np.full((max_n, 2), -1, dtype=np.float32)
        if graph.pos2d is not None:
            # 若graph.pos2d不足 actual_n 行，会自动覆盖
            pos2d_matrix[:actual_n, :] = graph.pos2d[:actual_n, :]

        return feature_matrix, adj_padded, mask, pos2d_matrix

    def get_train_data(self):
        return self.train_graphs

    def get_val_data(self):
        return self.val_graphs

    def get_test_data(self):
        return self.test_graphs