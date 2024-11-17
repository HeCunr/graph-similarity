import os
import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb  # 新增wandb导入
import time
from config.GF_config import gf_args
from model.layers.DenseGraphMatching import GraphMatchNetwork
from utils.GF_utils import (
    setup_logger,
    get_device,
    set_seed,
    cross_validation_split,
    generate_batches,
    generate_pairs_from_batch,
    prepare_batch_data,
    save_model,
    load_model,
    Graph
)
from utils.GF_early_stopping import EarlyStopping
from model.GF_dataset import GFDataset, GraphData

class GFTrainer:
    def __init__(self, args):
        self.args = args
        torch.set_default_tensor_type('torch.FloatTensor')
        self.device = get_device(args)

        # Setup logging
        self.log_dir, self.logger = setup_logger(args, "GF")
        self.logger.info(f"Using device: {self.device}")

        # Set random seed
        set_seed(args.seed)

        # Initialize dataset
        self.dataset = GFDataset(args.data_dir, args)
        self.logger.info(f"Loaded dataset with {len(self.dataset.graphs)} graphs")

        # Create model - 移动到wandb初始化之前
        self.model = GraphMatchNetwork(
            node_init_dims=args.graph_init_dim,
            args=args
        ).to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr
        )

        # Initialize wandb - 移到模型初始化之后
        self.init_wandb()
        # Create checkpoint paths
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def init_wandb(self):
        """Initialize Weights & Biases"""
        if self.args.disable_wandb:
            self.logger.info("Wandb logging is disabled")
            return

        try:
            settings = wandb.Settings(
                init_timeout=300,
                _disable_stats=True,
                _offline=False
            )

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

    def train_fold(
            self,
            train_indices: List[int],
            val_indices: List[int],
            fold: int
    ) -> float:
        """Train model on one fold"""
        self.logger.info(f"Training fold {fold + 1}/{self.args.n_folds}")

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            path=os.path.join(self.checkpoint_dir, f"fold_{fold+1}_checkpoint.pt")
        )

        pbar = tqdm(range(self.args.epochs), desc=f"Fold {fold+1}")
        for epoch in pbar:
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_graphs = [self.dataset.graphs[i] for i in train_indices]
            train_batches = generate_batches(train_graphs, self.args.batch_size)

            batch_losses = []  # 记录每个batch的损失
            batch_pbar = tqdm(train_batches, desc=f"Epoch {epoch+1}", leave=False)
            for batch_idx, batch in enumerate(batch_pbar):
                pairs = generate_pairs_from_batch(batch)
                if not pairs:
                    continue

                features1, adj1, features2, adj2 = prepare_batch_data(pairs, self.device)

                # Data augmentation and forward pass
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

                z1_view1 = self.model(drop_feature1, drop_edge1)
                z1_view2 = self.model(drop_feature1, drop_edge2)
                z2_view1 = self.model(drop_feature2, drop_edge1)
                z2_view2 = self.model(drop_feature2, drop_edge2)

                z1_orig = self.model(features1, adj1)
                z2_orig = self.model(features2, adj2)

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

                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                train_loss += batch_loss

                # 更新进度条
                batch_pbar.set_postfix({
                    'batch_loss': f'{batch_loss:.6f}'
                })

                # Log batch metrics to wandb
                wandb.log({
                    f"fold_{fold+1}/batch_loss": batch_loss,
                    f"fold_{fold+1}/batch_idx": batch_idx + epoch * len(train_batches)
                })

            # Calculate average training loss
            train_loss /= len(train_batches)

            # Validation phase
            val_loss = self.validate([self.dataset.graphs[i] for i in val_indices])

            # Log epoch metrics to wandb
            wandb.log({
                f"fold_{fold+1}/epoch": epoch,
                f"fold_{fold+1}/train_loss": train_loss,
                f"fold_{fold+1}/val_loss": val_loss,
                f"fold_{fold+1}/learning_rate": self.optimizer.param_groups[0]['lr']
            })

            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}'
            })

            # Logging
            self.logger.info(
                f"Fold {fold+1}, Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )

            # Early stopping
            if early_stopping(val_loss, self.model, epoch, self.optimizer):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                # Log early stopping to wandb
                wandb.log({
                    f"fold_{fold+1}/early_stopping_epoch": epoch
                })
                break

        # Load best model
        best_epoch, best_loss = early_stopping.load_checkpoint(self.model, self.optimizer)
        return best_loss

    def validate(self, graphs: List[GraphData]) -> float:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        val_batches = generate_batches(graphs, self.args.batch_size)

        with torch.no_grad():
            val_pbar = tqdm(val_batches, desc="Validating", leave=False)
            for batch in val_pbar:
                pairs = generate_pairs_from_batch(batch)
                if not pairs:
                    continue

                features1, adj1, features2, adj2 = prepare_batch_data(pairs, self.device)

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

                z1_view1 = self.model(drop_feature1, drop_edge1)
                z1_view2 = self.model(drop_feature1, drop_edge2)
                z2_view1 = self.model(drop_feature2, drop_edge1)
                z2_view2 = self.model(drop_feature2, drop_edge2)

                z1_orig = self.model(features1, adj1)
                z2_orig = self.model(features2, adj2)

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
                val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        return val_loss / len(val_batches)

    def train(self):
        """Main training procedure with cross-validation"""
        self.logger.info("Starting training with cross-validation...")
        cv_splits, test_indices = cross_validation_split(
            self.dataset.graphs,
            self.args.n_folds,
            self.args.test_split
        )

        cv_losses = []
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            self.logger.info(f"\nStarting fold {fold+1}/{self.args.n_folds}")
            self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

            best_val_loss = self.train_fold(train_idx, val_idx, fold)
            cv_losses.append(best_val_loss)

            wandb.log({
                f"cv_loss_fold_{fold+1}": best_val_loss
            })

            self.logger.info(f"Fold {fold+1} best validation loss: {best_val_loss:.6f}")

        mean_cv_loss = np.mean(cv_losses)
        std_cv_loss = np.std(cv_losses)

        wandb.log({
            "cv_mean_loss": mean_cv_loss,
            "cv_std_loss": std_cv_loss
        })

        self.logger.info(
            f"Cross-validation completed:\n"
            f"Mean validation loss: {mean_cv_loss:.6f} ± {std_cv_loss:.6f}"
        )

        # Final test evaluation
        self.logger.info("\nStarting final testing...")
        test_loss = self.validate([self.dataset.graphs[i] for i in test_indices])

        wandb.log({
            "test_loss": test_loss
        })

        self.logger.info(f"Final test loss: {test_loss:.6f}")

        return {
            "cv_mean": mean_cv_loss,
            "cv_std": std_cv_loss,
            "test_loss": test_loss,
            "cv_losses": cv_losses
        }

def main():
    trainer = GFTrainer(gf_args)
    results = trainer.train()

    # Log final summary metrics
    wandb.log({
        "final_cv_mean_loss": results['cv_mean'],
        "final_cv_std_loss": results['cv_std'],
        "final_test_loss": results['test_loss']
    })

    trainer.logger.info("Training completed!")
    trainer.logger.info(f"Cross-validation loss: {results['cv_mean']:.6f} ± {results['cv_std']:.6f}")
    trainer.logger.info(f"Test loss: {results['test_loss']:.6f}")

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()