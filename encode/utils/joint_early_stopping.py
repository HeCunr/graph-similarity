# utils/joint_early_stopping.py
import numpy as np
import torch
import os
from datetime import datetime

class JointEarlyStopping:
    def __init__(self, config):
        self.patience = config.patience
        self.min_delta = config.get('min_delta', 1e-4)  # 最小改进阈值
        self.counter = 0
        self.best_loss = None
        self.best_individual_losses = None
        self.early_stop = False
        self.model_dir = config.model_dir
        self.no_improvement_threshold = config.get('no_improvement_threshold', 3)  # 连续N轮无改进的阈值

        # 分别记录两个模型的最佳损失
        self.best_gf_loss = float('inf')
        self.best_dxf_loss = float('inf')

        # 记录改进的历史
        self.improvement_history = []
        os.makedirs(self.model_dir, exist_ok=True)

    def __call__(self, joint_loss, individual_losses, models, optimizer, epoch, fold=None):
        """
        早停检查
        Args:
            joint_loss: 联合损失值
            individual_losses: 包含各模型单独损失的字典
            models: 包含两个模型的字典
            optimizer: 优化器
            epoch: 当前轮次
            fold: 当前折数（可选）
        """
        loss = joint_loss if isinstance(joint_loss, float) else joint_loss.item()
        gf_loss = individual_losses['gf_loss']
        dxf_loss = individual_losses['dxf_loss']

        if self.best_loss is None:
            self.best_loss = loss
            self.best_gf_loss = gf_loss
            self.best_dxf_loss = dxf_loss
            self.best_individual_losses = individual_losses
            self.save_checkpoint(models, optimizer, loss, individual_losses, epoch, fold)
        else:
            # 检查是否有显著改进
            gf_improved = gf_loss < (self.best_gf_loss - self.min_delta)
            dxf_improved = dxf_loss < (self.best_dxf_loss - self.min_delta)
            joint_improved = loss < (self.best_loss - self.min_delta)

            if joint_improved or gf_improved or dxf_improved:
                # 记录改进
                if joint_improved:
                    self.best_loss = loss
                if gf_improved:
                    self.best_gf_loss = gf_loss
                if dxf_improved:
                    self.best_dxf_loss = dxf_loss

                self.best_individual_losses = individual_losses
                self.save_checkpoint(models, optimizer, loss, individual_losses, epoch, fold)
                self.counter = 0
                self.improvement_history.append(1)
            else:
                self.counter += 1
                self.improvement_history.append(0)
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

                # 检查最近N轮是否有任何改进
                recent_improvements = sum(self.improvement_history[-self.no_improvement_threshold:])
                if recent_improvements == 0 and len(self.improvement_history) >= self.no_improvement_threshold:
                    print(f"No improvement in last {self.no_improvement_threshold} epochs")

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, models, optimizer, loss, individual_losses, epoch, fold=None):
        """保存检查点，增加更多信息"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fold_str = f'_fold{fold}' if fold is not None else ''

        checkpoint = {
            'epoch': epoch,
            'gf_model_state_dict': models['gf'].state_dict(),
            'dxf_model_state_dict': models['dxf'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'joint_loss': loss,
            'individual_losses': individual_losses,
            'improvement_history': self.improvement_history
        }

        checkpoint_path = os.path.join(
            self.model_dir,
            f'checkpoint_{timestamp}{fold_str}.pt'
        )

        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

    def get_best_checkpoint(self):
        """获取最佳检查点信息"""
        return {
            'best_loss': self.best_loss,
            'best_individual_losses': self.best_individual_losses,
            'best_gf_loss': self.best_gf_loss,
            'best_dxf_loss': self.best_dxf_loss,
            'total_epochs': len(self.improvement_history)
        }

    def load_best_model(self, models, checkpoint_path):
        """加载最佳模型"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        models['gf'].load_state_dict(checkpoint['gf_model_state_dict'])
        models['dxf'].load_state_dict(checkpoint['dxf_model_state_dict'])
        return checkpoint['loss']