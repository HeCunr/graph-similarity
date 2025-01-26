# Geom_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from config.Geom_config import geom_args
from model.GeomLayers.DenseGeomMatching import GraphMatchNetwork
from dataset.Geom_dataset import GeomDataset
from utils.Geom_utils import (
    setup_logger,
    get_device,
    set_seed,
    generate_batches,
    generate_pairs_from_batch,
    prepare_batch_data
)
from utils.Geom_early_stopping import EarlyStopping
import math
from torch.optim.lr_scheduler import LambdaLR

from model.GeomLayers.NodeAggregator import MultiLevelNodeAggregator


class GeomTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args)
        self.log_dir, self.logger = setup_logger(args, "Geom")
        self.logger.info(f"Using device: {self.device}")

        set_seed(args.seed)

        # 初始化数据集，一次性得到 train / val / test
        self.dataset = GeomDataset(args.data_dir, args)
        self.train_graphs = self.dataset.get_train_data()
        self.val_graphs   = self.dataset.get_val_data()
        self.test_graphs  = self.dataset.get_test_data()

        # 多层节点聚合模块: 用于将原始大图(4096维)先聚合到128维等
        self.pooling_module = MultiLevelNodeAggregator(in_features=args.graph_init_dim).to(self.device)

        # 建立模型（只保留图匹配 + InfoNCE 等逻辑）
        self.model = GraphMatchNetwork(
            node_init_dims=args.graph_init_dim,
            args=args
        ).to(self.device)

        # 优化器 (注意把 pooling_module 的参数也加入)
        all_params = list(self.model.parameters()) + list(self.pooling_module.parameters())
        self.optimizer = optim.Adam(all_params, lr=args.lr)

        # 学习率调度器
        self.warmup_epochs = int(0.1 * args.epochs)
        self.total_epochs = args.epochs
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda
        )

        # 初始化 wandb
        self.init_wandb()

        # checkpoint 路径
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 初始化 EarlyStopping，需要指定保存路径
        self.early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            path=os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
        )

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

    def _lr_lambda(self, epoch):
        # warmup + cosine
        if epoch < self.warmup_epochs:
            return 0.1 + 0.9 * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * 0.9 + 0.1

    def train(self):
        """直接单次训练 + 验证 + 测试。"""
        self.logger.info("Starting single train/val/test procedure...")

        # 初始化早停
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

            # === 更新学习率 ===
            self.scheduler.step()

            # === 2) 验证 ===
            val_loss = self.validate(self.val_graphs)

            # 更新最佳验证损失和 epoch
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            # 日志
            self.logger.info(f"Epoch [{epoch+1}/{self.args.epochs}], train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # 3) 早停检查
            if early_stopping(
                    val_loss=val_loss,
                    model=self.model,
                    pooling_module=self.pooling_module,
                    epoch=epoch,
                    optimizer=self.optimizer
            ):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # === 加载最优模型，做最终测试 ===
        # 从早停实例中获取实际的最佳验证损失
        best_val_loss = early_stopping.val_loss_min
        early_stopping.load_checkpoint(
            model=self.model,
            pooling_module=self.pooling_module,
            optimizer=self.optimizer
        )
        test_loss = self.validate(self.test_graphs)
        self.logger.info(f"Final Test Loss: {test_loss:.4f}")
        wandb.log({"test_loss": test_loss})

        return {
            "best_epoch": best_epoch + 1,  # epoch 从 0 开始计数，需 +1
            "best_val_loss": best_val_loss,
            "test_loss": test_loss
        }

    def _train_one_epoch(self, train_batches):
        self.model.train()
        self.pooling_module.train()
        total_loss = 0.0

        for batch in tqdm(train_batches, desc="Training", leave=False):
            pairs = generate_pairs_from_batch(batch)
            if not pairs:
                continue

            features1, adj1, masks1, features2, adj2, masks2 = prepare_batch_data(pairs, self.device)

            # --- 1) 数据增强 ---
            drop_feature1 = self.model.drop_feature(features1, self.args.drop_feature1)
            drop_edge1 = torch.stack([
                self.model.aug_random_edge(a.cpu().numpy(), self.args.drop_edge1)
                for a in adj1
            ]).to(self.device)

            drop_feature2 = self.model.drop_feature(features2, self.args.drop_feature2)
            drop_edge2 = torch.stack([
                self.model.aug_random_edge(a.cpu().numpy(), self.args.drop_edge2)
                for a in adj2
            ]).to(self.device)

            # --- 2) 节点聚合 (4096->128)，这是必须保留的 NodeAggregator ---
            pfeat1, padj1, pmask1 = self.pooling_module(drop_feature1, drop_edge1, masks1)
            pfeat2, padj2, pmask2 = self.pooling_module(drop_feature2, drop_edge2, masks2)

            # --- 3) 送入 GNN (产生节点级特征 [B, N, d]) ---
            z1_view1 = self.model(pfeat1, padj1, pmask1)
            z1_view2 = self.model(pfeat1, padj2, pmask1)
            z2_view1 = self.model(pfeat2, padj1, pmask2)
            z2_view2 = self.model(pfeat2, padj2, pmask2)

            # 原特征视图 (不做drop) 也经过 pool + GNN，用于跨图匹配
            pfeat1_orig, padj1_orig, pmask1_orig = self.pooling_module(features1, adj1, masks1)
            z1_orig = self.model(pfeat1_orig, padj1_orig, pmask1_orig)

            pfeat2_orig, padj2_orig, pmask2_orig = self.pooling_module(features2, adj2, masks2)
            z2_orig = self.model(pfeat2_orig, padj2_orig, pmask2_orig)

            # --- 4) matching_layer (输出 [B, N, perspectives]) ---
            z1_view1, z1_view2 = self.model.matching_layer(z1_view1, z1_view2)
            z2_view1, z2_view2 = self.model.matching_layer(z2_view1, z2_view2)

            # 交叉匹配到对方图的原视图
            z1_view1, _ = self.model.matching_layer(z1_view1, z2_orig)
            z1_view2, _ = self.model.matching_layer(z1_view2, z2_orig)
            z2_view1, _ = self.model.matching_layer(z2_view1, z1_orig)
            z2_view2, _ = self.model.matching_layer(z2_view2, z1_orig)

            # === 在 InfoNCE 对比损失计算之前先进行归一化 (仅沿最后一维 perspectives 归一化) ===
            z1_view1 = nn.functional.normalize(z1_view1, dim=-1)
            z1_view2 = nn.functional.normalize(z1_view2, dim=-1)
            z2_view1 = nn.functional.normalize(z2_view1, dim=-1)
            z2_view2 = nn.functional.normalize(z2_view2, dim=-1)

            # --- 5) 计算 InfoNCE 损失（不再进行任何图级聚合，直接在 [B, N, P] 上算） ---
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
        self.pooling_module.eval()
        val_batches = generate_batches(val_graphs, self.args.batch_size)
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_batches:
                pairs = generate_pairs_from_batch(batch)
                if not pairs:
                    continue

                features1, adj1, masks1, features2, adj2, masks2 = prepare_batch_data(pairs, self.device)

                # 数据增强
                drop_feature1 = self.model.drop_feature(features1, self.args.drop_feature1)
                drop_edge1 = torch.stack([
                    self.model.aug_random_edge(a.cpu().numpy(), self.args.drop_edge1)
                    for a in adj1
                ]).to(self.device)

                drop_feature2 = self.model.drop_feature(features2, self.args.drop_feature2)
                drop_edge2 = torch.stack([
                    self.model.aug_random_edge(a.cpu().numpy(), self.args.drop_edge2)
                    for a in adj2
                ]).to(self.device)

                # 节点聚合
                pfeat1, padj1, pmask1 = self.pooling_module(drop_feature1, drop_edge1, masks1)
                pfeat2, padj2, pmask2 = self.pooling_module(drop_feature2, drop_edge2, masks2)

                # GNN
                z1_view1 = self.model(pfeat1, padj1, pmask1)
                z1_view2 = self.model(pfeat1, padj2, pmask1)
                z2_view1 = self.model(pfeat2, padj1, pmask2)
                z2_view2 = self.model(pfeat2, padj2, pmask2)

                # 原视图
                pfeat1_orig, padj1_orig, pmask1_orig = self.pooling_module(features1, adj1, masks1)
                z1_orig = self.model(pfeat1_orig, padj1_orig, pmask1_orig)

                pfeat2_orig, padj2_orig, pmask2_orig = self.pooling_module(features2, adj2, masks2)
                z2_orig = self.model(pfeat2_orig, padj2_orig, pmask2_orig)

                # matching
                z1_view1, z1_view2 = self.model.matching_layer(z1_view1, z1_view2)
                z2_view1, z2_view2 = self.model.matching_layer(z2_view1, z2_view2)

                # 交叉匹配
                z1_view1, _ = self.model.matching_layer(z1_view1, z2_orig)
                z1_view2, _ = self.model.matching_layer(z1_view2, z2_orig)
                z2_view1, _ = self.model.matching_layer(z2_view1, z1_orig)
                z2_view2, _ = self.model.matching_layer(z2_view2, z1_orig)

                # 归一化
                z1_view1 = nn.functional.normalize(z1_view1, dim=-1)
                z1_view2 = nn.functional.normalize(z1_view2, dim=-1)
                z2_view1 = nn.functional.normalize(z2_view1, dim=-1)
                z2_view2 = nn.functional.normalize(z2_view2, dim=-1)

                # 损失
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
    trainer = GeomTrainer(geom_args)
    results = trainer.train()
    trainer.logger.info("Training finished!")
    trainer.logger.info(f"Best epoch: {results['best_epoch']}")
    trainer.logger.info(f"Best val loss: {results['best_val_loss']:.4f}")
    trainer.logger.info(f"Test loss: {results['test_loss']:.4f}")

    wandb.finish()
    trainer.cleanup()


if __name__ == "__main__":
    main()
