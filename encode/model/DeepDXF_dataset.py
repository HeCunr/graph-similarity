# model/DeepDXF_dataset.py
import h5py
import torch
import os
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

class DXFDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            if 'dxf_vec' not in f:
                raise KeyError("Dataset 'dxf_vec' not found in the h5 file.")
            self.length = len(f['dxf_vec'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            dxf_vec = f['dxf_vec'][idx]

            # 检查数据有效性
            if np.any(np.isnan(dxf_vec)) or np.any(np.isinf(dxf_vec)):
                print(f"Warning: Invalid data found in {self.h5_file} at index {idx}")
                # 替换无效值为-1,-1表示没有该参数
                dxf_vec = np.where(np.isnan(dxf_vec) | np.isinf(dxf_vec), -1, dxf_vec)

        entity_type = torch.tensor(dxf_vec[:, 0], dtype=torch.long)
        entity_params = torch.tensor(dxf_vec[:, 1:], dtype=torch.float)
        return entity_type, entity_params

def load_h5_files(directory):
    datasets = []
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            try:
                dataset = DXFDataset(file_path)
                datasets.append(dataset)
            except KeyError as e:
                print(f"Error loading {filename}: {e}")
                continue
    if not datasets:
        raise ValueError("No valid datasets found in the specified directory.")
    return ConcatDataset(datasets)

# 确保在文件末尾添加这行
__all__ = ['DXFDataset', 'load_h5_files']