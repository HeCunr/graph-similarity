# model/joint_dataset.py
from torch.utils.data import DataLoader, SubsetRandomSampler
from model.GF_dataset import GFDataset
from model.DeepDXF_dataset import load_h5_files
from sklearn.model_selection import KFold
import numpy as np
import torch

# model/joint_dataset.py

# model/joint_dataset.py
class JointDataLoader:
    def __init__(self, config):
        self.config = config
        self.n_folds = config.n_folds
        self.test_size = config.test_size

        # 初始化数据集
        self.gf_dataset = GFDataset(data_dir=config.gf_data_dir,
                                    batch_size=config.gf_batch_size)
        self.dxf_dataset = load_h5_files(config.dxf_data_dir)

        # 初始化数据集划分
        # GF数据集不需要额外划分，因为GFDataset已经在内部完成了划分
        # 对DXF数据集进行划分
        dataset_size = len(self.dxf_dataset)
        indices = np.random.permutation(dataset_size)
        test_size = int(self.test_size * dataset_size)
        self.dxf_test_indices = indices[:test_size]
        self.dxf_train_indices = indices[test_size:]

    def _get_dxf_fold_loader(self, fold_idx, batch_size):
        """获取DeepDXF模型的fold数据加载器"""
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        train_folds = list(kfold.split(self.dxf_train_indices))

        # 获取当前折的训练和验证索引
        fold_train_indices = self.dxf_train_indices[train_folds[fold_idx][0]]
        fold_val_indices = self.dxf_train_indices[train_folds[fold_idx][1]]

        # 创建采样器
        train_sampler = SubsetRandomSampler(fold_train_indices)
        val_sampler = SubsetRandomSampler(fold_val_indices)

        # 创建数据加载器
        train_loader = DataLoader(
            self.dxf_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            self.dxf_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader

    def get_fold_loaders(self, fold_idx):
        """获取指定折的训练和验证数据加载器"""
        # GF数据
        fold_train_graphs, fold_val_graphs = self.gf_dataset.get_fold_data(fold_idx)

        # 创建GF数据加载器
        gf_train_loader = self.gf_dataset.generate_pairs(fold_train_graphs, self.config.gf_batch_size)
        gf_val_loader = self.gf_dataset.generate_pairs(fold_val_graphs, self.config.gf_batch_size)

        # 获取当前折的训练和验证索引
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        train_folds = list(kfold.split(self.dxf_train_indices))
        fold_train_indices = self.dxf_train_indices[train_folds[fold_idx][0]]
        fold_val_indices = self.dxf_train_indices[train_folds[fold_idx][1]]

        # 创建DXF数据加载器
        dxf_train_sampler = SubsetRandomSampler(fold_train_indices)
        dxf_val_sampler = SubsetRandomSampler(fold_val_indices)

        # 保证DXF数据加载器batch_size与GF同步
        dxf_batch_size = int(len(fold_train_indices) / len(gf_train_loader))

        dxf_train_loader = DataLoader(
            self.dxf_dataset,
            batch_size=dxf_batch_size,
            sampler=dxf_train_sampler,
            num_workers=4,
            pin_memory=True
        )

        dxf_val_loader = DataLoader(
            self.dxf_dataset,
            batch_size=dxf_batch_size,
            sampler=dxf_val_sampler,
            num_workers=4,
            pin_memory=True
        )

        return (gf_train_loader, gf_val_loader), (dxf_train_loader, dxf_val_loader)

    def get_test_loaders(self):
        """获取测试集数据加载器"""
        # GF测试集 - 使用特有的数据生成方法
        gf_test_loader = self.gf_dataset.generate_pairs(
            self.gf_dataset.test_graphs,
            self.config.gf_batch_size
        )

        # DeepDXF测试集
        dxf_test_loader = DataLoader(
            self.dxf_dataset,
            batch_size=self.config.dxf_batch_size,
            sampler=SubsetRandomSampler(self.dxf_test_indices),
            num_workers=4,
            pin_memory=True
        )

        return gf_test_loader, dxf_test_loader

    def get_data_stats(self):
        """获取数据集统计信息"""
        return {
            'gf_total': len(self.gf_dataset.graphs),
            'gf_train': len(self.gf_dataset.train_graphs),
            'gf_test': len(self.gf_dataset.test_graphs),
            'dxf_total': len(self.dxf_dataset),
            'dxf_train': len(self.dxf_train_indices),
            'dxf_test': len(self.dxf_test_indices)
        }