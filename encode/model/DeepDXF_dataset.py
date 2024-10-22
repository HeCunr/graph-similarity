
import h5py
import torch
from torch.utils.data import Dataset

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

        entity_type = torch.tensor(dxf_vec[:, 0], dtype=torch.long)
        entity_params = torch.tensor(dxf_vec[:, 1:], dtype=torch.float)
        print(f"entity_type shape: {entity_type.shape}")
        print(f"entity_params shape: {entity_params.shape}")

        return entity_type, entity_params