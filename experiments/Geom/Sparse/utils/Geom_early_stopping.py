# utils/Geom_early_stopping.py

import numpy as np
import torch
import os
from typing import Optional

class EarlyStopping:
    """
    简化版 EarlyStopping:
    仅根据 val_loss (越低越好) 进行保存和判停。
    不再保存/加载任何 pooling_module, 只保存当前 model/optimizer 状态。
    """
    def __init__(
            self,
            patience: int = 30,
            verbose: bool = False,
            delta: float = 0,
            path: str = 'checkpoints/Geom/BestSparseModel.pt',
            trace_func: Optional[callable] = print
    ):
        """
        Args:
            patience: 允许多少次 val_loss 未提升
            verbose: 是否打印提示
            delta: 最小提升阈值
            path: 保存的 checkpoint 路径
            trace_func: 打印函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        # 若需要可在此创建目录
        save_dir = os.path.dirname(self.path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int, optimizer: torch.optim.Optimizer) -> bool:
        """
        如果 val_loss 没有改善 -> 计数+1
        如果 val_loss 改善 -> 保存 & 重置计数
        超过 patience -> 早停
        """
        score = -val_loss  # 越小越好 => -val_loss 越大越好

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func:
                self.trace_func(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

        return self.early_stop

    def _save_checkpoint(self, val_loss: float, model: torch.nn.Module, epoch: int, optimizer: torch.optim.Optimizer):
        """
        仅保存 model/optimizer 的状态
        """
        if self.verbose and self.trace_func:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }
        torch.save(state, self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        加载最佳模型
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No checkpoint found at {self.path}")

        checkpoint = torch.load(self.path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def reset(self):
        """手动重置计数等状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
