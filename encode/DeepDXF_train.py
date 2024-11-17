import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split, SubsetRandomSampler
from model.DeepDXF_dataset import DXFDataset
from model.transformer_encoder import DXFTransformer
from model.DeepDXF_loss import DXFContrastiveLoss
from utils.DeepDXF_early_stopping import EarlyStopping
import h5py
from model.DeepDXF_dataset import load_h5_files
from config.DeepDXF_config import DXFConfig
import os
import argparse
import torch.cuda.amp as amp
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

import wandb
from torch.utils.data import DataLoader, ConcatDataset, random_split, SubsetRandomSampler

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


class DXFTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = amp.GradScaler()
        self.early_stopping = EarlyStopping(patience=cfg.patience)

        # 先初始化模型和损失函数
        self.init_model()

        # 然后初始化wandb
        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.wandb_name,
                config=vars(cfg),
                settings=wandb.Settings(start_method="fork")
            )
            # 记录模型架构
            wandb.watch(self.model)

    def init_model(self):
        """初始化或重置模型"""
        self.model = DXFTransformer(
            d_model=self.cfg.d_model,
            num_layers=self.cfg.num_layers,
            dim_z=self.cfg.dim_z,
            nhead=self.cfg.nhead,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.dropout,
            latent_dropout=self.cfg.latent_dropout
        ).to(self.device)

        self.contrastive_loss = DXFContrastiveLoss(
            cfg=self.cfg,
            device=self.device,
            batch_size=self.cfg.batch_size,
            temperature=self.cfg.temperature
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay  # 添加权重衰减
        )

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.cfg.epochs}')
        for batch_idx, (entity_type, entity_params) in enumerate(pbar):
            try:
                # 将数据移到设备上
                entity_type = entity_type.to(self.device)
                entity_params = entity_params.to(self.device)

                # 使用混合精度训练
                with amp.autocast():
                    outputs = self.model(entity_type, entity_params)
                    losses = self.contrastive_loss(outputs)
                    loss = losses["loss_contrastive"]

                # 记录每个batch的损失
                if self.cfg.use_wandb:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": batch_idx + epoch * len(dataloader)
                    })

                # 检查损失值是否是有限的
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected: {loss.item()}")
                    continue

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # 添加梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                num_batches += 1
                # 更新进度条显示
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': total_loss / num_batches
                })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        return total_loss / max(num_batches, 1) if num_batches > 0 else float('inf')

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for entity_type, entity_params in val_loader:
                try:
                    entity_type = entity_type.to(self.device)
                    entity_params = entity_params.to(self.device)

                    outputs = self.model(entity_type, entity_params)
                    losses = self.contrastive_loss(outputs)
                    loss = losses["loss_contrastive"]

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_batches += 1

                except Exception as e:
                    print(f"Error in validation: {str(e)}")
                    continue

        return total_loss / max(num_batches, 1) if num_batches > 0 else float('inf')

    def train_fold(self, train_loader, val_loader, fold_idx):
        print(f"Training Fold {fold_idx + 1}")

        # 重新初始化模型，确保每个fold从头开始训练
        self.init_model()

        best_val_loss = float('inf')
        fold_dir = os.path.join('checkpoints', f'fold_{fold_idx+1}')
        os.makedirs(fold_dir, exist_ok=True)


        # 记录每个fold的指标
        fold_metrics = []
        for epoch in range(self.cfg.epochs):
            try:
                train_loss = self.train_epoch(train_loader, epoch)
                val_loss = self.validate(val_loader)

                print(f"Epoch {epoch+1}/{self.cfg.epochs}")
                print(f"Training Loss: {train_loss:.4f}")
                print(f"Validation Loss: {val_loss:.4f}")

                # 记录到wandb
                if self.cfg.use_wandb:
                    wandb.log({
                        f"fold_{fold_idx+1}/train_loss": train_loss,
                        f"fold_{fold_idx+1}/val_loss": val_loss,
                        "epoch": epoch
                    })

                    # 可以添加更多的指标
                    learning_rate = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        f"fold_{fold_idx+1}/learning_rate": learning_rate,
                        "epoch": epoch
                    })

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(fold_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_val_loss,
                    }, checkpoint_path)

                    # 上传最佳模型到wandb
                    if self.cfg.use_wandb:
                        wandb.save(checkpoint_path)

                # 记录fold指标
                fold_metrics.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

                # Early stopping
                self.early_stopping(val_loss, self.model, self.optimizer, epoch)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue

        # 在fold结束时记录汇总指标
        if self.cfg.use_wandb:
            wandb.log({
                f"fold_{fold_idx+1}/best_val_loss": best_val_loss,
                f"fold_{fold_idx+1}/total_epochs": epoch + 1
            })

        return best_val_loss

    def test(self, test_loader):
        # 测试过程中的指标记录
        test_metrics = {}
        # 加载最佳模型
        best_val_loss = float('inf')
        best_fold = 0

        # 找到表现最好的fold
        for fold in range(self.cfg.n_folds):
            fold_path = os.path.join('checkpoints', f'fold_{fold+1}', 'best_model.pth')
            if os.path.exists(fold_path):
                checkpoint = torch.load(fold_path)
                if checkpoint['loss'] < best_val_loss:
                    best_val_loss = checkpoint['loss']
                    best_fold = fold + 1

        # 加载最佳fold的模型
        best_model_path = os.path.join('checkpoints', f'fold_{best_fold}', 'best_model.pth')
        print(f"Loading best model from fold {best_fold}")
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 测试
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for entity_type, entity_params in test_loader:
                try:
                    entity_type = entity_type.to(self.device)
                    entity_params = entity_params.to(self.device)

                    outputs = self.model(entity_type, entity_params)
                    losses = self.contrastive_loss(outputs)
                    loss = losses["loss_contrastive"]

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_batches += 1

                except Exception as e:
                    print(f"Error in testing: {str(e)}")
                    continue

        avg_test_loss = total_loss / max(num_batches, 1) if num_batches > 0 else float('inf')
        print(f"Test Loss: {avg_test_loss:.4f}")


        # 记录测试结果
        if self.cfg.use_wandb:
            wandb.log({
                "test_loss": avg_test_loss,
                "final_model_performance": avg_test_loss
            })
        return avg_test_loss

def main(args):
    cfg = DXFConfig(args)

    try:
        # wandb初始化配置
        if cfg.use_wandb:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name,
                config=vars(cfg)
            )
        # 加载数据
        data_dir = args.data_dir if args.data_dir else cfg.data_dir
        combined_dataset = load_h5_files(data_dir)

        # 划分测试集和训练集
        dataset_size = len(combined_dataset)
        test_size = int(0.2 * dataset_size)
        train_size = dataset_size - test_size

        train_dataset, test_dataset = random_split(
            combined_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建测试集数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # K折交叉验证
        k_folds = 10
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # 将数据集索引转换为列表以进行K折划分
        train_indices = list(range(train_size))

        fold_val_losses = []
        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_indices)):
            print(f'\nFOLD {fold+1}/{k_folds}')

            # 创建数据加载器
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)

            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )

            val_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )

            # 创建新的训练器实例
            trainer = DXFTrainer(cfg)

            # 训练当前折
            val_loss = trainer.train_fold(train_loader, val_loader, fold)
            fold_val_losses.append(val_loss)

        # 输出交叉验证结果
        print("\nCross-validation Results:")
        for fold, loss in enumerate(fold_val_losses):
            print(f"Fold {fold+1}: {loss:.4f}")
        print(f"Average validation loss: {np.mean(fold_val_losses):.4f}")

        # 在测试集上评估最佳模型
        test_loss = trainer.test(test_loader)
        print(f"\nFinal Test Loss: {test_loss:.4f}")
        # 在训练结束时关闭wandb
        if cfg.use_wandb:
            wandb.finish()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        if cfg.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DXF Transformer with contrastive loss')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing h5 files')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--loss_type', type=str, default='infonce', choices=['simclr', 'infonce'], help='Type of contrastive loss to use')
    # 添加wandb相关参数
    parser.add_argument('--wandb_project', type=str, default="DeepDXF", help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity(username)')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    args = parser.parse_args()
    main(args)