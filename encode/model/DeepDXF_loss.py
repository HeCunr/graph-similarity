import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.DeepDXF_utils import _get_padding_mask, _get_visibility_mask

class DXFContrastiveLoss(nn.Module):
    def __init__(self, cfg, device, batch_size, temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        # 直接访问属性，而不是使用get方法
        self.cl_loss_type = cfg.cl_loss_type
        self.weights = cfg.loss_weights

    def forward(self, outputs):
        if self.cl_loss_type == 'infonce':
            logits, labels = self._info_nce_loss(outputs["proj_z1"], outputs["proj_z2"])
            loss_contrastive = self.weights["loss_cl_weight"] * torch.nn.CrossEntropyLoss()(logits, labels)
        else:  # simclr
            loss_contrastive = self.weights['loss_cl_weight'] * \
                               self._contrastive_loss(outputs["proj_z1"], outputs["proj_z2"])

        return {"loss_contrastive": loss_contrastive}

    def _info_nce_loss(self, f1, f2):
        """
        计算 InfoNCE loss
        Args:
            f1: 第一个视图的特征 [batch_size, feature_dim]
            f2: 第二个视图的特征 [batch_size, feature_dim]
        Returns:
            logits: 相似度分数
            labels: 对应的标签
        """
        # 替换nan_to_num为手动实现
        def replace_nan_inf(x):
            x = x.clone()
            x[torch.isnan(x)] = 0.0
            x[torch.isinf(x)] = 0.0
            return x

        f1 = replace_nan_inf(f1)
        f2 = replace_nan_inf(f2)

        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        if not (torch.isfinite(f1).all() and torch.isfinite(f2).all()):
            print("Warning: Input features contain non-finite values")
            return torch.tensor(0.0, device=self.device), torch.tensor(0, device=self.device)

        batch_size = f1.size(0)

        # 特征归一化
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1, f2], dim=0)  # [2*batch_size, feature_dim]

        # 计算相似度矩阵 [2*batch_size, 2*batch_size]
        similarity_matrix = torch.matmul(features, features.T)

        # 创建标签矩阵
        labels = torch.zeros(2 * batch_size, 2 * batch_size, device=self.device)
        # 设置正样本对的位置
        labels[:batch_size, batch_size:] = torch.eye(batch_size, device=self.device)
        labels[batch_size:, :batch_size] = torch.eye(batch_size, device=self.device)

        # 移除对角线元素
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)

        # 获取非对角线元素
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        labels = labels[~mask].view(2 * batch_size, -1)

        # 选择正负样本对
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        # 应用温度系数
        logits = logits / self.temperature

        return logits, labels

    def _contrastive_loss(self, z1, z2):
        # 归一化
        z1 = F.normalize(z1.mean(1))  # (N, D)
        z2 = F.normalize(z2.mean(1))  # (N, D)

        batch_size = z1.size(0)
        labels = F.one_hot(torch.arange(batch_size), batch_size * 2).float().to(self.device)
        masks = F.one_hot(torch.arange(batch_size), batch_size).to(self.device)

        # 计算四种相似度矩阵
        logits_aa = torch.matmul(z1, z1.T) / self.temperature
        logits_aa = logits_aa - masks * 1e9
        logits_bb = torch.matmul(z2, z2.T) / self.temperature
        logits_bb = logits_bb - masks * 1e9
        logits_ab = torch.matmul(z1, z2.T) / self.temperature
        logits_ba = torch.matmul(z2, z1.T) / self.temperature

        # 计算损失
        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b).mean()
        return loss