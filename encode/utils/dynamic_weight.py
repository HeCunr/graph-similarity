# utils/dynamic_weight.py
import torch
import numpy as np
from collections import deque

class DynamicWeightAllocator:
    def __init__(self, method='uncertainty', temp=0.5, window_size=100, device=None):
        self.method = method
        self.temp = temp
        self.window_size = window_size
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        # 使用deque实现固定大小的滑动窗口
        self.loss_history = {
            'gf': deque(maxlen=window_size),
            'dxf': deque(maxlen=window_size)
        }
        self.grad_history = {
            'gf': deque(maxlen=window_size),
            'dxf': deque(maxlen=window_size)
        }

        # 记录最近的权重用于平滑，增加稳定性
        self.recent_weights = deque(maxlen=5)

        # 添加数值稳定性参数
        self.eps = 1e-8

    def update_history(self, losses, grads=None):
        """更新损失和梯度历史，添加安全检查"""
        # 确保损失值是有限的
        if torch.isfinite(losses[0]):
            self.loss_history['gf'].append(losses[0].item())
        if torch.isfinite(losses[1]):
            self.loss_history['dxf'].append(losses[1].item())

        if grads is not None:
            if torch.isfinite(grads[0]):
                self.grad_history['gf'].append(grads[0].norm().item())
            if torch.isfinite(grads[1]):
                self.grad_history['dxf'].append(grads[1].norm().item())

    def compute_weights(self, losses, grads=None):
        """计算动态权重，增加安全检查和数值稳定性"""
        self.update_history(losses, grads)

        if self.method == 'uncertainty':
            weights = self._uncertainty_weights(losses)
        elif self.method == 'grad_norm':
            weights = self._grad_based_weights(grads)
        elif self.method == 'loss_ratio':
            weights = self._loss_ratio_weights()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 确保权重是标量
        if isinstance(weights, torch.Tensor) and len(weights.shape) > 1:
            weights = weights.mean(dim=0)

        # 应用温度缩放
        weights = self._apply_temperature(weights)

        # 平滑权重变化
        weights = self._smooth_weights(weights)

        # 确保权重非负且和为1
        weights = torch.abs(weights)
        weights = weights / (weights.sum() + self.eps)

        return weights.to(self.device)

    def _uncertainty_weights(self, losses):
        """基于不确定度的权重计算，添加安全检查"""
        # 安全地计算损失的移动平均
        loss_means = {
            'gf': np.mean(self.loss_history['gf']) if self.loss_history['gf'] else losses[0].item(),
            'dxf': np.mean(self.loss_history['dxf']) if self.loss_history['dxf'] else losses[1].item()
        }

        # 安全地计算损失的方差
        loss_vars = {
            'gf': np.var(list(self.loss_history['gf'])) if len(self.loss_history['gf']) > 1 else 0,
            'dxf': np.var(list(self.loss_history['dxf'])) if len(self.loss_history['dxf']) > 1 else 0
        }

        # 结合均值和方差计算权重，添加数值稳定性
        uncertainties = torch.tensor([
            1.0 / (max(loss_means['gf'], self.eps) + max(loss_vars['gf'], self.eps)),
            1.0 / (max(loss_means['dxf'], self.eps) + max(loss_vars['dxf'], self.eps))
        ])

        return torch.softmax(uncertainties / self.temp, dim=0)

    def _grad_based_weights(self, grads):
        """基于梯度范数的权重计算"""
        if not grads:
            return torch.tensor([0.5, 0.5])

        # 计算梯度范数的移动平均
        recent_grads_gf = np.mean(list(self.grad_history['gf'])[-10:])
        recent_grads_dxf = np.mean(list(self.grad_history['dxf'])[-10:])

        # 计算相对梯度强度
        total_grad = recent_grads_gf + recent_grads_dxf + 1e-8
        weights = torch.tensor([
            recent_grads_gf / total_grad,
            recent_grads_dxf / total_grad
        ])

        return weights

    def _loss_ratio_weights(self):
        """基于损失比例的权重计算"""
        if not self.loss_history['gf']:
            return torch.tensor([0.5, 0.5])

        # 计算损失的移动平均
        recent_loss_gf = np.mean(list(self.loss_history['gf'])[-10:])
        recent_loss_dxf = np.mean(list(self.loss_history['dxf'])[-10:])

        # 计算相对损失比例
        total_loss = recent_loss_gf + recent_loss_dxf + 1e-8
        weights = torch.tensor([
            recent_loss_gf / total_loss,
            recent_loss_dxf / total_loss
        ])

        return weights

    def _apply_temperature(self, weights):
        """应用温度缩放"""
        weights = weights.pow(1.0 / self.temp)
        return weights / weights.sum()

    def _smooth_weights(self, weights):
        """平滑权重变化，增加稳定性"""
        self.recent_weights.append(weights)
        if len(self.recent_weights) > 0:
            # 使用指数移动平均
            alpha = 0.7
            smoothed_weights = weights.clone()
            for prev_weights in reversed(list(self.recent_weights)[:-1]):
                smoothed_weights = alpha * smoothed_weights + (1 - alpha) * prev_weights
            return smoothed_weights
        return weights

    def get_weight_stats(self):
        """获取权重统计信息，增加监控"""
        stats = {
            'gf_loss_mean': np.mean(list(self.loss_history['gf'])) if self.loss_history['gf'] else 0,
            'dxf_loss_mean': np.mean(list(self.loss_history['dxf'])) if self.loss_history['dxf'] else 0,
            'gf_loss_var': np.var(list(self.loss_history['gf'])) if len(self.loss_history['gf']) > 1 else 0,
            'dxf_loss_var': np.var(list(self.loss_history['dxf'])) if len(self.loss_history['dxf']) > 1 else 0,
            'gf_grad_mean': np.mean(list(self.grad_history['gf'])) if self.grad_history['gf'] else 0,
            'dxf_grad_mean': np.mean(list(self.grad_history['dxf'])) if self.grad_history['dxf'] else 0
        }
        return stats