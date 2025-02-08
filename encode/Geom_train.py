# Geom_train.py

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

from config.Geom_config import geom_args
from dataset.Geom_dataset import GeomDataset
# 引入新的 collate_graphs
from utils.Geom_utils import get_device, set_seed, collate_graphs, drop_feature, aug_random_edge,drop_pos2d
from utils.Geom_early_stopping import EarlyStopping, os
from model.GeomLayers.GeomModel import GeomModel
from model.GeomLayers.GeomAlignment import NodeAlignmentHead

class GeomTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args)
        set_seed(args.seed)

        # 数据集
        self.dataset = GeomDataset(args.data_dir, args)
        self.train_graphs = self.dataset.get_train_data()
        self.val_graphs = self.dataset.get_val_data()
        self.test_graphs = self.dataset.get_test_data()

        self.train_loader = DataLoader(
            self.train_graphs, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs
        )
        self.val_loader = DataLoader(
            self.val_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs
        )
        self.test_loader = DataLoader(
            self.test_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs
        )

        # 核心模型
        self.model = GeomModel(
            args=args,
            d_model=256
        ).to(self.device)

        # 对齐 + 节点级对比损失
        self.alignment_head = NodeAlignmentHead(
            d_model=256,
            alignment=args.alignment,
            perspectives=args.perspectives,
            tau=args.tau
        ).to(self.device)

        # 优化器
        all_params = list(self.model.parameters()) + list(self.alignment_head.parameters())
        self.optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

        # 学习率调度
        self.warmup_epochs = int(0.1 * args.epochs)
        self.total_epochs = args.epochs
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_scale
        )

        # 如果需要 W&B
        if not args.disable_wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args)
            )

        os.makedirs("checkpoints", exist_ok=True)
        self.early_stopping = EarlyStopping(
            patience=args.patience,
            path="checkpoints/Geom/N3_Layer1_GGNN.pt"
        )

    def _lr_scale(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.1 + 0.9 * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * 0.9 + 0.1

    def train(self):
        best_val_loss = float('inf')
        best_epoch = -1

        for epoch in range(self.args.epochs):
            train_loss = self._train_one_epoch()
            val_loss = self.evaluate(self.val_loader)
            self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            print(f"Epoch [{epoch+1}/{self.args.epochs}], train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if not self.args.disable_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if self.early_stopping(val_loss, self.model, epoch, self.optimizer):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 加载最优模型
        self.early_stopping.load_checkpoint(self.model, self.optimizer)
        test_loss = self.evaluate(self.test_loader)
        print(f"[Best epoch={best_epoch+1}] final test_loss={test_loss:.4f}")

        if not self.args.disable_wandb:
            wandb.log({"test_loss": test_loss})
            wandb.finish()

    def _train_one_epoch(self):
        self.model.train()
        self.alignment_head.train()
        total_loss = 0.0

        # 添加进度条
        for (features, adjs, masks, pos2ds, graph_names) in tqdm(self.train_loader, desc="Train Epoch", leave=False):
            features = features.to(self.device)
            adjs = adjs.to(self.device)
            masks = masks.to(self.device)
            pos2ds = pos2ds.to(self.device)

            # 数据增强
            f1 = drop_feature(features.clone(), self.args.drop_feature1)
            f2 = drop_feature(features.clone(), self.args.drop_feature2)
            p1 = drop_pos2d(pos2ds.clone(), self.args.drop_pos1)
            p2 = drop_pos2d(pos2ds.clone(), self.args.drop_pos2)
            a1 = aug_random_edge(adjs.clone(), self.args.drop_edge1)
            a2 = aug_random_edge(adjs.clone(), self.args.drop_edge2)

            x1, _, _ = self.model(f1, p1, a1, masks)
            x2, _, _ = self.model(f2, p2, a2, masks)

            z1, z2 = self.alignment_head.perform_alignment(x1, x2)
            loss = self.alignment_head.loss(z1, z2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        self.alignment_head.eval()
        total_loss = 0.0

        # 这里同样使用新的 collate 函数处理批次
        for (features, adjs, masks, pos2ds, graph_names) in tqdm(loader, desc="Eval", leave=False):
            features = features.to(self.device)
            adjs     = adjs.to(self.device)
            masks    = masks.to(self.device)
            pos2ds   = pos2ds.to(self.device)

            # 这里若想禁用增广，则直接用features/pos2ds/adjs；若想和训练保持一致可继续drop
            f1 = drop_feature(features.clone(), self.args.drop_feature1)
            f2 = drop_feature(features.clone(), self.args.drop_feature2)
            p1 = drop_pos2d(pos2ds.clone(), self.args.drop_pos1)
            p2 = drop_pos2d(pos2ds.clone(), self.args.drop_pos2)
            a1 = aug_random_edge(adjs.clone(), self.args.drop_edge1)
            a2 = aug_random_edge(adjs.clone(), self.args.drop_edge2)

            x1, _, _ = self.model(f1, p1, a1, masks)
            x2, _, _ = self.model(f2, p2, a2, masks)
            z1, z2 = self.alignment_head.perform_alignment(x1, x2)
            loss = self.alignment_head.loss(z1, z2)

            total_loss += loss.item()

        return total_loss / len(loader)


def main():
    trainer = GeomTrainer(geom_args)
    trainer.train()

if __name__ == "__main__":
    main()
