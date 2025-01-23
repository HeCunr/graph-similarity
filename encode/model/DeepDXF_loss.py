import torch
import torch.nn as nn
import torch.nn.functional as F

class DXFContrastiveLoss(nn.Module):
    def __init__(self, cfg, device, batch_size, temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size  # 训练时你传入 N (实际上可以不用)
        self.weights = cfg.loss_weights

    def forward(self, outputs):
        """
        outputs 包含:
          - proj_z1: shape (N,64,256)
          - proj_z2: shape (N,64,256)
        返回一个 dict, 其中 'loss_contrastive' 是 InfoNCE 损失
        """
        logits, labels = self._info_nce_loss(outputs["proj_z1"], outputs["proj_z2"])
        # 这里 logits.shape = [2*N*S, 1+negatives], labels.shape=[2*N*S]
        loss_contrastive = self.weights["loss_cl_weight"] * torch.nn.CrossEntropyLoss()(logits, labels)
        return {"loss_contrastive": loss_contrastive}

    def _info_nce_loss(self, f1, f2):
        """
        f1, f2: (N,64,256), 其中 N=batch_size, 64=序列长度, 256=通道。
        先 flatten -> (N*64,256) => 再拼到 (2*N*64,256) 做 InfoNCE，
        得到正对 i 与 i+N*64。
        """
        def replace_nan_inf(x):
            x = x.clone()
            x[torch.isnan(x)] = 0.0
            x[torch.isinf(x)] = 0.0
            return x

        # => (N*S,256)
        N, S, D = f1.shape
        f1 = f1.reshape(N*S, D)
        f2 = f2.reshape(N*S, D)

        f1 = replace_nan_inf(f1)
        f2 = replace_nan_inf(f2)

        # 归一化
        f1 = F.normalize(f1, dim=1)  # => (N*S,256)
        f2 = F.normalize(f2, dim=1)  # => (N*S,256)

        # 合并 => (2*N*S,256)
        features = torch.cat([f1, f2], dim=0)
        total_size = features.size(0)  # 2*N*S

        # 相似度矩阵 => (2*N*S, 2*N*S)
        similarity_matrix = torch.matmul(features, features.T)

        # 构建标签
        # i与i+(N*S)是正样本对, 其余都是负样本
        # 这里 batch_size_for_label = N*S
        # positives: i <-> i+ (N*S)
        label_matrix = torch.zeros(total_size, total_size, device=self.device)
        # 前半段 f1 对应 0..(N*S-1)
        # 与 后半段 f2 对应 N*S..(2*N*S-1)
        # 构造单位阵
        eye_mat = torch.eye(N*S, device=self.device)
        label_matrix[:N*S, N*S:] = eye_mat
        label_matrix[N*S:, :N*S] = eye_mat

        # 移除对角线 (自己对自己)
        mask = torch.eye(total_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix[~mask].view(total_size, -1)
        label_matrix = label_matrix[~mask].view(total_size, -1)

        # positives => (2*N*S, 1)  每行只有1个正例
        positives = similarity_matrix[label_matrix.bool()].view(total_size, -1)
        # negatives => (2*N*S, 2*N*S-2)
        negatives = similarity_matrix[~label_matrix.bool()].view(total_size, -1)

        # 拼接: logit = [pos|neg] => shape=(2*N*S, 1 + #neg)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)  # 都是0, 表示第0列是positives

        # 温度缩放
        logits = logits / self.temperature
        return logits, labels
