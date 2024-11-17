# model/joint_loss.py
import torch
import torch.nn as nn
from model.layers.DenseGraphMatching import HierarchicalGraphMatchNetwork
from model.DeepDXF_loss import DXFContrastiveLoss
from utils.dynamic_weight import DynamicWeightAllocator

# model/joint_loss.py
class JointLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device  # 添加这行来存储device

        # 初始化子模型的损失函数
        self.gf_loss = HierarchicalGraphMatchNetwork(
            node_init_dims=config.gf_config.graph_init_dim,
            arguments=config.gf_config,
            device=config.device
        )

        self.dxf_loss = DXFContrastiveLoss(
            cfg=config.dxf_config,
            device=config.device,
            batch_size=config.dxf_batch_size,
            temperature=config.dxf_config.temperature
        )

        # 初始化动态权重分配器
        self.weight_allocator = DynamicWeightAllocator(
            method=config.weight_method,
            temp=config.temp,
            window_size=config.window_size,
            device=self.device
        )
        # 添加损失缩放因子来平衡量级
        self.gf_scale = config.get('gf_loss_scale', 0.1)
        self.dxf_scale = config.get('dxf_loss_scale', 0.1)

        # 添加数值稳定性的参数
        self.eps = 1e-8
        # 把权重分配器移到正确的设备上
        self.to(self.device)

    def _safe_mean(self, loss_tensor):
        """安全地计算平均值，处理可能的数值不稳定性"""
        if not torch.isfinite(loss_tensor).all():
            print(f"Warning: Non-finite values in loss tensor: {loss_tensor}")
            # 只使用有限值计算平均值
            finite_mask = torch.isfinite(loss_tensor)
            if finite_mask.any():
                return loss_tensor[finite_mask].mean()
            return torch.tensor(1e-8, device=self.device)
        return loss_tensor.mean()

    def forward(self, gf_outputs, dxf_outputs, return_individual=False):
        try:
            # 1. 计算GF损失
            feature_p1 = gf_outputs['feature_p1']
            feature_p2 = gf_outputs['feature_p2']
            feature_h1 = gf_outputs['feature_h1']
            feature_h2 = gf_outputs['feature_h2']

            loss_p = self.gf_loss.loss(feature_p1, feature_p2, batch_size=0)
            loss_h = self.gf_loss.loss(feature_h1, feature_h2, batch_size=0)
            gf_loss = (loss_p + loss_h) * 0.5
            gf_loss = self._safe_mean(gf_loss) * self.gf_scale
            # 2. 计算并检查DXF损失
            dxf_losses = self.dxf_loss(dxf_outputs)
            dxf_loss = dxf_losses["loss_contrastive"]
            dxf_loss = self._safe_mean(dxf_loss) * self.dxf_scale

            # 确保损失非零
            if gf_loss < 1e-8:
                gf_loss = torch.tensor(1e-8, device=self.device)
            if dxf_loss < 1e-8:
                dxf_loss = torch.tensor(1e-8, device=self.device)

            # 3. 梯度检查和记录
            if self.training:
                gf_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.gf_loss.parameters(),
                    max_norm=1.0
                )
                dxf_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.dxf_loss.parameters(),
                    max_norm=1.0
                )
                if not torch.isfinite(gf_grad_norm) or not torch.isfinite(dxf_grad_norm):
                    print(f"Warning: Non-finite gradients detected - GF: {gf_grad_norm}, DXF: {dxf_grad_norm}")

            # 4. 安全的权重计算
            losses = [
                torch.clamp(gf_loss, -100, 100),
                torch.clamp(dxf_loss, -100, 100)
            ]
            weights = self.weight_allocator.compute_weights(losses)

            weights = weights / (weights.sum() + self.eps)

            # 5. 计算联合损失
            joint_loss = weights[0] * gf_loss + weights[1] * dxf_loss

            # 6. 损失验证
            if not torch.isfinite(joint_loss):
                print(f"Warning: Non-finite joint loss detected: {joint_loss}")
                print(f"Individual losses - GF: {gf_loss}, DXF: {dxf_loss}")
                print(f"Weights: {weights}")
                joint_loss = torch.tensor(1.0, device=self.device)

            if return_individual:
                return joint_loss, {
                    'gf_loss': gf_loss.item(),
                    'dxf_loss': dxf_loss.item(),
                    'weights': weights.detach().cpu().numpy(),
                    'gf_grad_norm': gf_grad_norm.item() if self.training else 0,
                    'dxf_grad_norm': dxf_grad_norm.item() if self.training else 0
                }
            return joint_loss

        except Exception as e:
            print(f"Error in joint loss calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回一个可以反向传播的损失
            return torch.tensor(1e-8, requires_grad=True, device=self.device)

    def get_loss_stats(self):
        """获取损失统计信息"""
        stats = self.weight_allocator.get_weight_stats()
        stats.update({
            'gf_scale': self.gf_scale,
            'dxf_scale': self.dxf_scale,
        })
        return stats