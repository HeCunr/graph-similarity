# Seq_train.py
import torch
from torch.utils.data import DataLoader, random_split
import os
import argparse
import numpy as np
import traceback
from tqdm import tqdm
import wandb

from utils.Seq_augment import augment_seq_sample
from dataset.Seq_dataset import load_h5_files
from model.SeqLayers.seq_transformer_encoder import SeqTransformer
from model.SeqLayers.Seq_loss import SeqContrastiveLoss
from utils.Seq_early_stopping import EarlyStopping
from config.Seq_config import SeqConfig

# ========== 新增一个函数，用于对一个 batch 做两份增强并拆分 ==========

def two_augmentations_for_batch(entity_type_t, entity_params_t, device):
    """
    对同一个 batch (B,4096), (B,4096,43) 做两份增强：
    返回:
      aug1_type, aug1_param  => (B,4096), (B,4096,43)
      aug2_type, aug2_param  => (B,4096), (B,4096,43)
    """
    B = entity_type_t.size(0)

    # 先转回 CPU numpy，逐样本做 augment_seq_sample
    entity_type_np = entity_type_t.cpu().numpy().astype(np.int32)      # (B,4096)
    entity_params_np = entity_params_t.cpu().numpy().astype(np.int32)  # (B,4096,43)

    aug1_types_list = []
    aug1_params_list = []
    aug2_types_list = []
    aug2_params_list = []

    for i in range(B):
        combined_arr = np.zeros((4096, 44), dtype=np.int32)
        combined_arr[:, 0] = entity_type_np[i, :]
        combined_arr[:, 1:] = entity_params_np[i, :, :]

        # 分别做两份增强
        aug1_44 = augment_seq_sample(combined_arr)
        aug2_44 = augment_seq_sample(combined_arr)

        # 拆分 => type列+param列
        aug1_type = aug1_44[:, 0]
        aug1_param = aug1_44[:, 1:]
        aug2_type = aug2_44[:, 0]
        aug2_param = aug2_44[:, 1:]

        aug1_types_list.append(aug1_type)
        aug1_params_list.append(aug1_param)
        aug2_types_list.append(aug2_type)
        aug2_params_list.append(aug2_param)

    # 拼回 (B,4096), (B,4096,43)
    aug1_type_t = torch.from_numpy(np.stack(aug1_types_list, axis=0)).long().to(device)
    aug1_param_t = torch.from_numpy(np.stack(aug1_params_list, axis=0)).long().to(device)

    aug2_type_t = torch.from_numpy(np.stack(aug2_types_list, axis=0)).long().to(device)
    aug2_param_t = torch.from_numpy(np.stack(aug2_params_list, axis=0)).long().to(device)

    return aug1_type_t, aug1_param_t, aug2_type_t, aug2_param_t


class SeqTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # 设备
        if torch.cuda.is_available():
            if hasattr(cfg, 'gpu_id'):
                self.device = torch.device(f"cuda:{cfg.gpu_id}")
            else:
                self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")
            print("Warning: CUDA not available, using CPU.")

        os.makedirs('checkpoints', exist_ok=True)
        self.early_stopping = EarlyStopping(patience=cfg.patience)
        self.init_model()

        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.wandb_name,
                config=vars(cfg)
            )
            wandb.watch(self.model)

    def init_model(self):
        self.model = SeqTransformer(
            d_model=self.cfg.d_model,
            num_layers=self.cfg.num_layers,
            dim_z=self.cfg.dim_z,
            nhead=self.cfg.nhead,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.dropout,
            latent_dropout=self.cfg.latent_dropout
        ).to(self.device)

        self.contrastive_loss = SeqContrastiveLoss(
            cfg=self.cfg,
            device=self.device,
            batch_size=self.cfg.batch_size,
            temperature=self.cfg.temperature
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay
        )

        import math
        warmup_epochs = self.cfg.warmup_epochs
        total_epochs  = self.cfg.epochs

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                ratio = float(current_epoch) / float(warmup_epochs)
                factor = 1.0 + (self.cfg.max_lr / self.cfg.initial_lr - 1.0)*ratio
                return factor
            else:
                progress = (current_epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
                cos_out  = 0.5*(1.0 + math.cos(math.pi * progress))
                start_scale = self.cfg.max_lr / self.cfg.initial_lr
                end_scale   = self.cfg.final_lr / self.cfg.initial_lr
                factor = end_scale + (start_scale - end_scale)*cos_out
                return factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )

    def train(self, train_loader, val_loader, test_loader):
        best_val_loss = float('inf')

        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.cfg.epochs}")
            print(f"  Training Loss:   {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Learning Rate:   {current_lr:.6f}")

            if self.cfg.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch
                })

            # 保存 best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'checkpoints/best_model.pth')

                if self.cfg.use_wandb:
                    wandb.save('checkpoints/best_model.pth')

            self.early_stopping(val_loss, self.model, self.optimizer, epoch)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

        test_loss = self.test(test_loader)
        print(f"\nFinal Test Loss: {test_loss:.4f}")

        if self.cfg.use_wandb:
            wandb.log({"test_loss": test_loss})
            wandb.finish()

        return test_loss

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.cfg.epochs}')
        for batch_idx, (entity_type, entity_params) in enumerate(pbar):
            try:
                entity_type = entity_type.to(self.device)       # (B,4096)
                entity_params = entity_params.to(self.device)   # (B,4096,43)

                # -- 做两份增强 --
                aug1_type, aug1_param, aug2_type, aug2_param = two_augmentations_for_batch(
                    entity_type, entity_params, self.device
                )
                # 分别前向
                proj_z1 = self.model(aug1_type, aug1_param)  # => (B,64,256)
                proj_z2 = self.model(aug2_type, aug2_param)  # => (B,64,256)

                outputs = {"proj_z1": proj_z1, "proj_z2": proj_z2}
                losses = self.contrastive_loss(outputs)
                loss = losses["loss_contrastive"]

                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss: {loss.item()}")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': total_loss / num_batches,
                    'lr': self.scheduler.get_last_lr()[0]
                })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (entity_type, entity_params) in enumerate(val_loader):
                try:
                    entity_type = entity_type.to(self.device)
                    entity_params = entity_params.to(self.device)

                    aug1_type, aug1_param, aug2_type, aug2_param = two_augmentations_for_batch(
                        entity_type, entity_params, self.device
                    )
                    proj_z1 = self.model(aug1_type, aug1_param)
                    proj_z2 = self.model(aug2_type, aug2_param)

                    outputs = {"proj_z1": proj_z1, "proj_z2": proj_z2}
                    losses = self.contrastive_loss(outputs)
                    loss = losses["loss_contrastive"]

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_batches += 1

                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue

        return total_loss / max(num_batches, 1)

    def test(self, test_loader):
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (entity_type, entity_params) in enumerate(test_loader):
                try:
                    entity_type = entity_type.to(self.device)
                    entity_params = entity_params.to(self.device)

                    aug1_type, aug1_param, aug2_type, aug2_param = two_augmentations_for_batch(
                        entity_type, entity_params, self.device
                    )
                    proj_z1 = self.model(aug1_type, aug1_param)
                    proj_z2 = self.model(aug2_type, aug2_param)

                    outputs = {"proj_z1": proj_z1, "proj_z2": proj_z2}
                    losses = self.contrastive_loss(outputs)
                    loss = losses["loss_contrastive"]

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_batches += 1

                except Exception as e:
                    print(f"Error in test batch {batch_idx}: {str(e)}")
                    continue

        avg_test_loss = total_loss / max(num_batches, 1)
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss


def main(args):
    cfg = SeqConfig(args)

    try:
        data_dir = args.data_dir if args.data_dir else cfg.data_dir
        combined_dataset = load_h5_files(data_dir)

        total_size = len(combined_dataset)
        train_size = int(cfg.train_ratio * total_size)
        val_size = int(cfg.val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            combined_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        trainer = SeqTrainer(cfg)
        test_loss = trainer.train(train_loader, val_loader, test_loader)
        print(f"\nTraining completed. Final test loss: {test_loss:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        if cfg.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SeqTransformer with InfoNCE loss')
    parser.add_argument('--data_dir', type=str, default=r"/home/vllm/encode/data/Seq/TRAIN_4096", help='Directory containing h5 files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use')

    parser.add_argument('--wandb_project', type=str, default="Seq", help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity(username)')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    args = parser.parse_args()
    main(args)
