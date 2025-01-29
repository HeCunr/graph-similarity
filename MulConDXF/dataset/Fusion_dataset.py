# dataset/Fusion_dataset.py
import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FusionDataset(Dataset):
    def __init__(self, geom_dir, seq_dir, max_nodes=4096, max_seq_len=4096):
        super().__init__()
        self.geom_dir = geom_dir
        self.seq_dir  = seq_dir
        self.max_nodes = max_nodes
        self.max_seq_len = max_seq_len

        # 1) 收集所有 json 文件名（去掉后缀）
        geom_bases = []
        for f in os.listdir(geom_dir):
            if f.endswith('.json'):
                base = os.path.splitext(f)[0]
                geom_bases.append(base)

        # 2) 收集所有 h5 文件名
        seq_bases = []
        for f in os.listdir(seq_dir):
            if f.endswith('.h5'):
                base = os.path.splitext(f)[0]
                seq_bases.append(base)

        # 3) 求交集
        self.common_bases = sorted(set(geom_bases).intersection(seq_bases))
        self.length = len(self.common_bases)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base = self.common_bases[idx]
        json_path = os.path.join(self.geom_dir, base + '.json')
        h5_path   = os.path.join(self.seq_dir,  base + '.h5')

        # ---- 读取 geom json (未增强) ----
        with open(json_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            data = json.loads(line)
        n_num = data['n_num']
        features = np.array(data['features'], dtype=np.float32)  # [n_num, 44]
        succs = data['succs']

        # 构造 pad 后的 adjacency, feature, mask
        geom_feat = np.full((self.max_nodes, 44), -1, dtype=np.float32)
        geom_adj  = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        geom_mask = np.zeros((self.max_nodes,), dtype=np.float32)

        actual_n = min(n_num, self.max_nodes)
        geom_feat[:actual_n,:] = features[:actual_n,:]
        for u in range(actual_n):
            for v in succs[u]:
                if v<actual_n:
                    geom_adj[u,v] = 1
                    geom_adj[v,u] = 1
        np.fill_diagonal(geom_adj[:actual_n,:actual_n], 1)
        geom_mask[:actual_n] = 1

        # ---- 读取 seq h5 (未增强) ----
        with h5py.File(h5_path, 'r') as hf:
            seq_data = hf['data'][:]   # [4096,44] 假设已固定
        # 如果需要 mask 之类，也可自己构造

        return {
            "geom_feat": geom_feat,   # [4096,44]
            "geom_adj":  geom_adj,    # [4096,4096]
            "geom_mask": geom_mask,   # [4096]
            "seq_data":  seq_data,    # [4096,44] (示例)
            "filename":  base
        }


def fusion_collate_fn(batch):
    geom_feats = []
    geom_adjs  = []
    geom_masks = []
    seq_datas  = []
    filenames  = []
    for item in batch:
        geom_feats.append(torch.tensor(item['geom_feat'], dtype=torch.float32))
        geom_adjs.append( torch.tensor(item['geom_adj'],  dtype=torch.float32))
        geom_masks.append(torch.tensor(item['geom_mask'], dtype=torch.float32))
        seq_datas.append( torch.tensor(item['seq_data'],  dtype=torch.float32))
        filenames.append(item['filename'])

    geom_feats = torch.stack(geom_feats, dim=0)  # [B,4096,44]
    geom_adjs  = torch.stack(geom_adjs,  dim=0)  # [B,4096,4096]
    geom_masks = torch.stack(geom_masks, dim=0)  # [B,4096]
    seq_datas  = torch.stack(seq_datas,  dim=0)  # [B,4096,44]

    return {
        "geom_feat": geom_feats,
        "geom_adj":  geom_adjs,
        "geom_mask": geom_masks,
        "seq_data":  seq_datas,
        "filename":  filenames
    }

def get_fusion_dataloader(geom_dir, seq_dir, batch_size=8, shuffle=True):
    dataset = FusionDataset(geom_dir, seq_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=fusion_collate_fn)
    return loader
