import torch
import json
import numpy as np
from model.layers.DenseGraphMatching import GraphMatchNetwork
from config.GF_config import gf_args
from utils.GF_utils import get_device, Graph, prepare_batch_data

class GraphSimilarityCalculator:
    def __init__(self, model_path):
        self.device = get_device(gf_args)
        self.model = GraphMatchNetwork(
            node_init_dims=gf_args.graph_init_dim,
            args=gf_args
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.max_nodes = gf_args.graph_size_max

    def load_graph(self, file_path):
        with open(file_path) as f:
            g_info = json.loads(f.readline().strip())

        graph = Graph(g_info['n_num'])
        graph.features = np.array(g_info['features'])

        # Add edges
        for i in range(g_info['n_num']):
            for j in g_info['succs'][i]:
                graph.add_edge(i, j)

        # Process graph
        feature_matrix = np.zeros((self.max_nodes, gf_args.graph_init_dim))
        if len(graph.features.shape) == 1:
            graph.features = graph.features.reshape(1, -1)
        feature_matrix[:g_info['n_num'], :] = graph.features

        adj_matrix = graph.get_adjacency_matrix()
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
        adj_padded = np.zeros((self.max_nodes, self.max_nodes))
        adj_padded[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix

        return feature_matrix, adj_padded

    def compute_similarity(self, graph1_path, graph2_path):
        feat1, adj1 = self.load_graph(graph1_path)
        feat2, adj2 = self.load_graph(graph2_path)

        feat1 = torch.FloatTensor(feat1).unsqueeze(0).to(self.device)
        adj1 = torch.FloatTensor(adj1).unsqueeze(0).to(self.device)
        feat2 = torch.FloatTensor(feat2).unsqueeze(0).to(self.device)
        adj2 = torch.FloatTensor(adj2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z1 = self.model(feat1, adj1)
            z2 = self.model(feat2, adj2)

            z1_matched, z2_matched = self.model.matching_layer(z1, z2)

            z1_norm = torch.norm(z1_matched, p=2, dim=-1)
            z2_norm = torch.norm(z2_matched, p=2, dim=-1)
            sim = torch.sum(z1_matched * z2_matched) / (torch.sum(z1_norm * z2_norm) + 1e-8)

            sim = (sim + 1) / 2

        return sim.item()

if __name__ == "__main__":
    model_path = r"/home/vllm/encode/pretrained/GF/bacth_size_128/checkpoints/final_model.pt"
    calculator = GraphSimilarityCalculator(model_path)

    # Example graph paths
    graph1_path = r"/home/vllm/encode/cl_data/discreted_dataset/TEST/QFN28LK(Cu)-90-450 Rev1_3.json"
    graph2_path = r"/home/vllm/encode/cl_data/discreted_dataset/TEST/QFN28LK(Cu)-90-450 Rev1_4.json"

    similarity = calculator.compute_similarity(graph1_path, graph2_path)
    print(f"graph1_path and graph2_path : Graph similarity: {graph1_path} and {graph2_path} : {similarity:.4f}")

