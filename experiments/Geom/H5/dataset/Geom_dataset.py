#dataset/Geom_dataset.py
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class GraphData:
    """
    用来存储单个图的数据结构，包含:
      self.node_num
      self.name
      self.matrices = (feature_matrix, adj_padded, mask, pos2d_matrix)
    """
    def __init__(self, node_num: int, name: str):
        self.node_num = node_num
        self.name = name
        self.matrices = None

class GeomDataset(Dataset):
    """
    按题意：这里从 data_dir 下所有的 '.h5' 文件中，加载每个图的数据。
    然后再拆分 train/val/test。
    """
    def __init__(self, data_dir: str, args):
        self.args = args
        self.data_dir = data_dir

        # 1) 找到所有 h5 文件
        all_files = os.listdir(data_dir)
        self.h5_files = [f for f in all_files if f.endswith('.h5')]

        self.graphs = []
        for hf in self.h5_files:
            full_path = os.path.join(data_dir, hf)
            with h5py.File(full_path, 'r') as f:
                src_str = f['src'][()]    # 读单个标量字符串
                n_num   = int(f['n_num'][()])

                feat = f['feature_matrix'][()]   # numpy array
                adj  = f['adj_padded'][()]
                msk  = f['mask'][()]
                pos2d= f['pos2d_matrix'][()]

            # 构造 GraphData
            if isinstance(src_str, bytes):
                src_str = src_str.decode('utf-8')

            g = GraphData(n_num, src_str)
            g.matrices = (feat, adj, msk, pos2d)
            self.graphs.append(g)

        self.num_graphs = len(self.graphs)

        # 2) 这里拆分 train/val/test
        indices = np.arange(self.num_graphs)
        np.random.seed(args.seed)
        np.random.shuffle(indices)

        train_end = int(self.num_graphs * args.train_split)
        val_end = int(self.num_graphs * (args.train_split + args.val_split))

        self.train_indices = indices[:train_end]
        self.val_indices   = indices[train_end:val_end]
        self.test_indices  = indices[val_end:]

        self.train_graphs = [self.graphs[i] for i in self.train_indices]
        self.val_graphs   = [self.graphs[i] for i in self.val_indices]
        self.test_graphs  = [self.graphs[i] for i in self.test_indices]

    def get_train_data(self):
        return self.train_graphs

    def get_val_data(self):
        return self.val_graphs

    def get_test_data(self):
        return self.test_graphs
