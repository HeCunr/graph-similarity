import copy
import json
import os
from typing import List, Tuple, Dict
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
import random

class GraphData:
    """Class to store a single graph's data"""
    def __init__(self, node_num: int, name: str = None):
        self.node_num = node_num
        self.name = name
        self.features = np.zeros((node_num, 0))  # Will be set later
        self.adj = nx.Graph()
        self.matrices = None  # Will store (feature_matrix, adj_matrix, mask)

    def add_edge(self, u: int, v: int):
        self.adj.add_edge(u, v)

class GFDataset:
    def __init__(self, data_dir: str, args):
        """
        Initialize GF dataset
        
        Args:
            data_dir: Directory containing the graph data
            args: Arguments from GF_config
        """
        self.args = args
        self.data_dir = data_dir

        # First load raw graphs without processing
        self.graphs = self.load_raw_graphs()
        self.num_graphs = len(self.graphs)

        # Calculate max nodes
        self.max_nodes = self._get_max_nodes()

        # Now process all graphs with known max_nodes
        self.graphs = self.process_loaded_graphs()

        # Split into test and train
        self.test_size = int(self.num_graphs * args.test_split)
        self.train_size = self.num_graphs - self.test_size

        # Random shuffle for splitting
        np.random.seed(args.seed)
        self.graph_indices = np.random.permutation(self.num_graphs)

        # Split indices
        self.test_indices = self.graph_indices[:self.test_size]
        self.train_indices = self.graph_indices[self.test_size:]

        # Initialize cross validation
        self.kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    def load_raw_graphs(self) -> List[GraphData]:
        """Load graphs without processing"""
        graphs = []
        print(f"Loading graphs from {self.data_dir}")

        for file in os.listdir(self.data_dir):
            if not file.endswith('.json'):
                continue

            with open(os.path.join(self.data_dir, file)) as f:
                for line in f:
                    g_info = json.loads(line.strip())
                    graph = GraphData(node_num=g_info['n_num'], name=g_info['src'])

                    # Set node features
                    graph.features = np.array(g_info['features'])

                    # Add edges
                    for u in range(g_info['n_num']):
                        for v in g_info['succs'][u]:
                            graph.add_edge(u, v)

                    graphs.append(graph)

        print(f"Loaded {len(graphs)} raw graphs")
        return graphs

    def process_loaded_graphs(self) -> List[GraphData]:
        """Process all loaded graphs"""
        processed_graphs = []
        print("Processing graphs...")

        for i, graph in enumerate(self.graphs):
            try:
                feature_matrix, adj_matrix, mask = self._process_single_graph(graph)
                graph_copy = copy.deepcopy(graph)
                graph_copy.matrices = (feature_matrix, adj_matrix, mask)
                processed_graphs.append(graph_copy)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(self.graphs)} graphs")

            except Exception as e:
                print(f"Error processing graph {graph.name}: {str(e)}")
                print(f"Graph info: nodes={graph.node_num}, features shape={graph.features.shape}")
                raise

        print("Graph processing completed")
        return processed_graphs

    def read_graphs(self) -> List[GraphData]:
        """Read all graphs from the data directory"""
        graphs = []
        print(f"Reading graphs from {self.data_dir}")  # 添加调试信息

        for file in os.listdir(self.data_dir):
            if not file.endswith('.json'):
                continue

            with open(os.path.join(self.data_dir, file)) as f:
                for line in f:
                    g_info = json.loads(line.strip())
                    graph = GraphData(node_num=g_info['n_num'], name=g_info['fname'])

                    # Set node features
                    graph.features = np.array(g_info['features'])

                    # Add edges
                    for u in range(g_info['n_num']):
                        for v in g_info['succs'][u]:
                            graph.add_edge(u, v)

                    # Process the graph immediately
                    feature_matrix, adj_matrix, mask = self._process_single_graph(graph)
                    graph.matrices = (feature_matrix, adj_matrix, mask)
                    graphs.append(graph)

        print(f"Loaded {len(graphs)} graphs")  # 添加调试信息
        return graphs

    def _get_max_nodes(self) -> int:
        """Get maximum number of nodes across all graphs"""
        return max(g.node_num for g in self.graphs)

    def preprocess_graphs(self) -> List[GraphData]:
        """Preprocess all graphs to have consistent dimensions"""
        processed = []
        print("Starting graph preprocessing...")  # 添加调试信息

        for i, g in enumerate(self.graphs):
            if g.matrices is None:  # 如果还没有处理过
                feature_matrix, adj_matrix, mask = self._process_single_graph(g)
                g_copy = copy.deepcopy(g)
                g_copy.matrices = (feature_matrix, adj_matrix, mask)
                processed.append(g_copy)
            else:
                processed.append(copy.deepcopy(g))

            if (i + 1) % 100 == 0:  # 添加进度信息
                print(f"Processed {i+1}/{len(self.graphs)} graphs")

        print("Graph preprocessing completed")
        return processed

    def _process_single_graph(self, graph: GraphData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single graph to have consistent dimensions"""
        try:
            # Prepare feature matrix
            feature_dim = self.args.graph_init_dim
            feature_matrix = np.zeros((self.max_nodes, feature_dim))
            if len(graph.features.shape) == 1:  # 如果特征是一维的，reshape它
                graph.features = graph.features.reshape(1, -1)
            feature_matrix[:graph.node_num, :] = graph.features

            # Prepare adjacency matrix
            adj_matrix = np.array(nx.to_numpy_matrix(graph.adj))
            adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])  # Add self-loops
            adj_padded = np.zeros((self.max_nodes, self.max_nodes))
            adj_padded[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix

            # Prepare mask
            mask = np.zeros(self.max_nodes)
            mask[:graph.node_num] = 1

            return feature_matrix, adj_padded, mask
        except Exception as e:
            print(f"Error processing graph {graph.name}: {str(e)}")
            print(f"Graph info: nodes={graph.node_num}, features shape={graph.features.shape}")
            raise

    def get_train_fold(self, fold_idx: int) -> Tuple[List[int], List[int]]:
        """Get train and validation indices for a specific fold"""
        fold_splits = list(self.kf.split(self.train_indices))
        train_idx, val_idx = fold_splits[fold_idx]
        return self.train_indices[train_idx], self.train_indices[val_idx]

    def get_test_data(self) -> List[GraphData]:
        """Get test set graphs"""
        return [self.processed_graphs[i] for i in self.test_indices]

    def generate_batch(self, indices: List[int], batch_size: int = None) -> List[Tuple[GraphData, GraphData]]:
        """Generate batches of graph pairs"""
        if batch_size is None:
            batch_size = self.args.batch_size

        # Shuffle indices
        indices = indices.copy()
        random.shuffle(indices)

        # Generate pairs
        pairs = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size:
                continue

            # Randomly pair graphs in the batch
            batch_indices = batch_indices.copy()
            random.shuffle(batch_indices)

            for j in range(0, batch_size, 2):
                idx1, idx2 = batch_indices[j], batch_indices[j+1]
                pairs.append((self.processed_graphs[idx1], self.processed_graphs[idx2]))

        return pairs

    def get_batch_tensors(self, pairs: List[Tuple[GraphData, GraphData]]) -> Tuple[torch.Tensor, ...]:
        """Convert a batch of graph pairs to tensors"""
        feat1_list, adj1_list = [], []
        feat2_list, adj2_list = [], []

        for g1, g2 in pairs:
            feat1, adj1, _ = g1.matrices
            feat2, adj2, _ = g2.matrices

            feat1_list.append(feat1)
            adj1_list.append(adj1)
            feat2_list.append(feat2)
            adj2_list.append(adj2)

        # Convert to tensors
        feat1_tensor = torch.FloatTensor(np.stack(feat1_list))
        adj1_tensor = torch.FloatTensor(np.stack(adj1_list))
        feat2_tensor = torch.FloatTensor(np.stack(feat2_list))
        adj2_tensor = torch.FloatTensor(np.stack(adj2_list))

        return feat1_tensor, adj1_tensor, feat2_tensor, adj2_tensor