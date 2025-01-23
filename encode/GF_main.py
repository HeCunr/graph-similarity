# GF_main.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb

from config.GF_config import gf_args
from model.layers.DenseGraphMatching import GraphMatchNetwork
from model.GF_dataset import GFDataset, GraphData
from utils.GF_utils import (
    setup_logger,
    get_device,
    set_seed,
    generate_batches,
    generate_pairs_from_batch,
    prepare_batch_data
)
from utils.GF_early_stopping import EarlyStopping

class GFTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args)
        self.log_dir, self.logger = setup_logger(args, "GF")
        self.logger.info(f"Using device: {self.device}")

        set_seed(args.seed)

        # 初始化数据集，一次性得到 train / val / test
        self.dataset = GFDataset(args.data_dir, args)
        self.train_graphs = self.dataset.get_train_data()
        self.val_graphs   = self.dataset.get_val_data()
        self.test_graphs  = self.dataset.get_test_data()

        # 建立模型
        self.model = GraphMatchNetwork(
            node_init_dims=args.graph_init_dim,
            args=args
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # 初始化 wandb
        self.init_wandb()

        # checkpoint 路径
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def init_wandb(self):
        if self.args.disable_wandb:
            self.logger.info("Wandb logging is disabled")
            return
        try:
            settings = wandb.Settings(init_timeout=300, _disable_stats=True, _offline=False)
            config = {
                "learning_rate": self.args.lr,
                "batch_size": self.args.batch_size,
                "epochs": self.args.epochs,
                "architecture": self.args.conv,
                "dataset": self.args.dataset,
                "graph_init_dim": self.args.graph_init_dim,
                "filters": self.args.filters,
                "dropout": self.args.dropout,
                "patience": self.args.patience,
                "model_parameters": sum(p.numel() for p in self.model.parameters())
            }
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.wandb_run_name,
                config=config,
                settings=settings
            )
            wandb.watch(self.model, log="all", log_freq=self.args.wandb_log_freq)
        except Exception as e:
            self.logger.error(f"Error initializing wandb: {str(e)}")
            self.logger.warning("Continuing training without wandb logging")

    def train(self):
        """不再做交叉验证和超参搜索，直接单次训练 + 验证 + 测试。"""
        self.logger.info("Starting single train/val/test procedure...")

        # 提前创建 early stopping
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            path=os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
        )

        train_batches = generate_batches(self.train_graphs, self.args.batch_size)
        best_val_loss = float('inf')
        best_epoch = -1

        for epoch in range(self.args.epochs):
            # === 1) 训练 ===
            train_loss = self._train_one_epoch(train_batches)

            # === 2) 验证 ===
            val_loss = self.validate(self.val_graphs)

            # 日志
            self.logger.info(f"Epoch [{epoch+1}/{self.args.epochs}], train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # 早停判断
            if early_stopping(val_loss, self.model, epoch, self.optimizer):
                best_val_loss = val_loss
                best_epoch = epoch
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # === 加载最优模型，做最终测试 ===
        early_stopping.load_checkpoint(self.model, self.optimizer)
        test_loss = self.validate(self.test_graphs)
        self.logger.info(f"Final Test Loss: {test_loss:.4f}")
        wandb.log({"test_loss": test_loss})

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss
        }

    def _train_one_epoch(self, train_batches):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(train_batches, desc="Training", leave=False):
            pairs = generate_pairs_from_batch(batch)
            if not pairs:
                continue

            features1, adj1, masks1, features2, adj2, masks2 = prepare_batch_data(pairs, self.device)

            # 做各种 data augmentation
            drop_feature1 = self.model.drop_feature(features1, self.args.drop_feature1)
            drop_edge1 = torch.stack([
                self.model.aug_random_edge(adj.cpu().numpy(), self.args.drop_edge1)
                for adj in adj1
            ]).to(self.device)

            drop_feature2 = self.model.drop_feature(features2, self.args.drop_feature2)
            drop_edge2 = torch.stack([
                self.model.aug_random_edge(adj.cpu().numpy(), self.args.drop_edge2)
                for adj in adj2
            ]).to(self.device)

            z1_view1 = self.model(drop_feature1, drop_edge1, masks1)
            z1_view2 = self.model(drop_feature1, drop_edge2, masks1)
            z2_view1 = self.model(drop_feature2, drop_edge1, masks2)
            z2_view2 = self.model(drop_feature2, drop_edge2, masks2)

            z1_orig = self.model(features1, adj1, masks1)
            z2_orig = self.model(features2, adj2, masks2)

            z1_view1, z1_view2 = self.model.matching_layer(z1_view1, z1_view2)
            z2_view1, z2_view2 = self.model.matching_layer(z2_view1, z2_view2)

            z1_view1, _ = self.model.matching_layer(z1_view1, z2_orig)
            z1_view2, _ = self.model.matching_layer(z1_view2, z2_orig)
            z2_view1, _ = self.model.matching_layer(z2_view1, z1_orig)
            z2_view2, _ = self.model.matching_layer(z2_view2, z1_orig)

            loss1 = self.model.loss(z1_view1, z1_view2, batch_size=0)
            loss2 = self.model.loss(z2_view1, z2_view2, batch_size=0)
            loss = (loss1 + loss2) * 0.5

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_batches)

    def validate(self, val_graphs):
        self.model.eval()
        val_batches = generate_batches(val_graphs, self.args.batch_size)
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_batches:
                pairs = generate_pairs_from_batch(batch)
                if not pairs:
                    continue

                features1, adj1, masks1, features2, adj2, masks2 = prepare_batch_data(pairs, self.device)
                # 与训练时类似
                drop_feature1 = self.model.drop_feature(features1, self.args.drop_feature1)
                drop_edge1 = torch.stack([
                    self.model.aug_random_edge(adj.cpu().numpy(), self.args.drop_edge1)
                    for adj in adj1
                ]).to(self.device)

                drop_feature2 = self.model.drop_feature(features2, self.args.drop_feature2)
                drop_edge2 = torch.stack([
                    self.model.aug_random_edge(adj.cpu().numpy(), self.args.drop_edge2)
                    for adj in adj2
                ]).to(self.device)

                z1_view1 = self.model(drop_feature1, drop_edge1, masks1)
                z1_view2 = self.model(drop_feature1, drop_edge2, masks2)
                z2_view1 = self.model(drop_feature2, drop_edge1, masks1)
                z2_view2 = self.model(drop_feature2, drop_edge2, masks2)

                z1_orig = self.model(features1, adj1, masks1)
                z2_orig = self.model(features2, adj2, masks2)

                z1_view1, z1_view2 = self.model.matching_layer(z1_view1, z1_view2)
                z2_view1, z2_view2 = self.model.matching_layer(z2_view1, z2_view2)

                z1_view1, _ = self.model.matching_layer(z1_view1, z2_orig)
                z1_view2, _ = self.model.matching_layer(z1_view2, z2_orig)
                z2_view1, _ = self.model.matching_layer(z2_view1, z1_orig)
                z2_view2, _ = self.model.matching_layer(z2_view2, z1_orig)

                loss1 = self.model.loss(z1_view1, z1_view2, batch_size=0)
                loss2 = self.model.loss(z2_view1, z2_view2, batch_size=0)
                loss = (loss1 + loss2) * 0.5

                val_loss += loss.item()

        return val_loss / len(val_batches)

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self, 'dataset'):
            del self.dataset


def main():
    trainer = GFTrainer(gf_args)
    results = trainer.train()
    trainer.logger.info("Training finished!")
    trainer.logger.info(f"Best epoch: {results['best_epoch']}")
    trainer.logger.info(f"Best val loss: {results['best_val_loss']:.4f}")
    trainer.logger.info(f"Test loss: {results['test_loss']:.4f}")

    # 关闭 wandb 等
    wandb.finish()
    trainer.cleanup()

if __name__ == "__main__":
    main()