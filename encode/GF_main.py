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
import itertools
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
        self.device = get_device(args)

        # Setup logging
        self.log_dir, self.logger = setup_logger(args, "GF")
        self.logger.info(f"Using device: {self.device}")

        # Set random seed
        set_seed(args.seed)

        # Initialize dataset
        self.dataset = GFDataset(args.data_dir, args)

        # Create model - moved before wandb init
        self.model = GraphMatchNetwork(
            node_init_dims=args.graph_init_dim,
            args=args
        ).to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr
        )

        # Initialize wandb
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
            fold: int,
            params: dict
    ) -> Tuple[float, dict]:
        """
        Train model on one fold with given hyperparameters

        Args:
            train_indices: Indices for training data
            val_indices: Indices for validation data
            fold: Current fold number
            params: Hyperparameters to use

        Returns:
            Tuple of (best_val_loss, metrics)
        """
        self.logger.info(f"Training fold {fold + 1}/{self.args.n_folds}")

        # Apply hyperparameters
        for param, value in params.items():
            setattr(self.args, param, value)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            path=os.path.join(self.checkpoint_dir, f"fold_{fold+1}_checkpoint.pt")
        )

        metrics = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }

        pbar = tqdm(range(self.args.epochs), desc=f"Fold {fold+1}")
        for epoch in pbar:
            # Training phase
            train_loss = self._train_epoch(train_indices)
            metrics['train_losses'].append(train_loss)

            # Validation phase
            val_loss = self.validate([self.dataset.graphs[i] for i in val_indices])
            metrics['val_losses'].append(val_loss)

            # Log to wandb
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

            # Early stopping
            if early_stopping(val_loss, self.model, epoch, self.optimizer):
                metrics['best_epoch'] = epoch
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        best_epoch, best_loss = early_stopping.load_checkpoint(self.model, self.optimizer)
        return best_loss, metrics

    def train(self):
        """Main training procedure with hyperparameter search and final evaluation"""
        self.logger.info("Starting training procedure...")

        # Split data using modified cross_validation_split
        try:
            cv_splits, train_val_indices, test_indices = cross_validation_split(
                self.dataset.graphs,
                self.args.n_folds,
                self.args.test_split,
                self.args.seed
            )

            self.logger.info(f"Dataset split complete: {len(train_val_indices)} train+val, {len(test_indices)} test")

            # Define hyperparameter search space
            param_grid = {
                'lr': [0.0001, 0.001],
                'dropout': [0.1, 0.3],
                'tau': [0.5, 0.7]
            }

            # Hyperparameter search
            best_params = None
            best_cv_loss = float('inf')
            param_results = []

            for params in self._get_param_combinations(param_grid):
                self.logger.info(f"\nTrying parameters: {params}")
                cv_losses = []

                for fold, (train_idx, val_idx) in enumerate(cv_splits):
                    # Reset model for each fold
                    self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                    self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])

                    # Train fold
                    fold_loss, metrics = self.train_fold(train_idx, val_idx, fold, params)
                    cv_losses.append(fold_loss)

                    # Log fold results
                    wandb.log({
                        f"fold_{fold+1}/best_val_loss": fold_loss,
                        f"fold_{fold+1}/params": params
                    })

                mean_cv_loss = np.mean(cv_losses)
                std_cv_loss = np.std(cv_losses)

                param_results.append({
                    'params': params,
                    'mean_loss': mean_cv_loss,
                    'std_loss': std_cv_loss,
                    'fold_losses': cv_losses
                })

                # Update best parameters if needed
                if mean_cv_loss < best_cv_loss:
                    best_cv_loss = mean_cv_loss
                    best_params = params

            # Log hyperparameter search results
            self._log_param_search_results(param_results)

            # Train final model with best parameters
            self.logger.info(f"\nTraining final model with best parameters: {best_params}")
            final_results = self._train_final_model(
                train_val_indices,
                test_indices,
                best_params
            )

            return {
                'best_params': best_params,
                'best_cv_loss': best_cv_loss,
                'final_test_loss': final_results['test_loss'],
                'cv_mean': np.mean([r['mean_loss'] for r in param_results]),
                'cv_std': np.std([r['mean_loss'] for r in param_results])
            }

        except Exception as e:
            self.logger.error(f"Error in training process: {str(e)}")
            raise

    def _train_final_model(self, train_val_indices: List[int], test_indices: List[int], params: dict) -> dict:
        """
        Train the final model on the complete training set with best parameters

        Args:
            train_val_indices: Indices for complete training set
            test_indices: Indices for test set
            params: Best hyperparameters from cross-validation

        Returns:
            dict: Final training results
        """
        # Apply best parameters
        for param, value in params.items():
            setattr(self.args, param, value)

        # Reset model for final training
        self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])

        # Train on full training set
        final_model_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            path=final_model_path
        )

        final_metrics = {
            'train_losses': [],
            'best_epoch': 0,
            'test_loss': 0.0
        }

        # Final training loop
        for epoch in range(self.args.epochs):
            train_loss = self._train_epoch(train_val_indices)
            final_metrics['train_losses'].append(train_loss)

            # Log metrics
            wandb.log({
                'final_training/epoch': epoch,
                'final_training/train_loss': train_loss
            })

            if early_stopping(train_loss, self.model, epoch, self.optimizer):
                final_metrics['best_epoch'] = epoch
                break

        # Load best model for testing
        early_stopping.load_checkpoint(self.model, self.optimizer)

        # Final evaluation on test set
        test_loss = self.validate([self.dataset.graphs[i] for i in test_indices])
        final_metrics['test_loss'] = test_loss

        self.logger.info(f"Final test loss: {test_loss:.6f}")
        wandb.log({'final_test_loss': test_loss})

        return final_metrics

    def _train_epoch(self, indices: List[int]) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        train_graphs = [self.dataset.graphs[i] for i in indices]
        train_batches = generate_batches(train_graphs, self.args.batch_size)

        for batch in tqdm(train_batches, desc="Training", leave=False):
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

            total_loss += loss.item()

        return total_loss / len(train_batches)

    def _get_param_combinations(self, param_grid: dict) -> List[dict]:
        """Generate all combinations of hyperparameters"""
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations

    def _log_param_search_results(self, results: List[dict]):
        """Log hyperparameter search results"""
        self.logger.info("\nHyperparameter search results:")
        for result in results:
            self.logger.info(
                f"Params: {result['params']}, "
                f"Mean loss: {result['mean_loss']:.6f} ± {result['std_loss']:.6f}"
            )

        wandb.log({
            'hparam_search/results': wandb.Table(
                data=[[str(r['params']), r['mean_loss'], r['std_loss']] for r in results],
                columns=['Parameters', 'Mean Loss', 'Std Loss']
            )
        })

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

    def cleanup(self):
        """Clear all GPU memory and resources"""
        # 删除模型
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'optimizer'):
            del self.optimizer

        # 清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 清理数据集
        if hasattr(self, 'dataset'):
            del self.dataset.graphs
            del self.dataset

def main():
    try:
        trainer = GFTrainer(gf_args)
        results = trainer.train()

        # Log final summary metrics
        wandb.log({
            "final_cv_mean_loss": results['cv_mean'],
            "final_cv_std_loss": results['cv_std'],
            "final_test_loss": results['final_test_loss']
        })

        trainer.logger.info("Training completed!")
        trainer.logger.info(f"Cross-validation loss: {results['cv_mean']:.6f} ± {results['cv_std']:.6f}")
        trainer.logger.info(f"Test loss: {results['final_test_loss']:.6f}")

    except Exception as e:
        trainer.logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        # 清理资源
        trainer.cleanup()

        # 关闭 wandb
        wandb.finish()

        # 确保所有 GPU 缓存被清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 显式调用垃圾回收
        import gc
        gc.collect()

if __name__ == "__main__":
    main()