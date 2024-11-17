# joint_train.py
import torch
import torch.cuda.amp as amp
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
import itertools  # 添加 itertools 导入
from model.layers.DenseGraphMatching import HierarchicalGraphMatchNetwork
from model.transformer_encoder import DXFTransformer
from model.joint_loss import JointLoss
from utils.joint_early_stopping import JointEarlyStopping
import traceback

class JointTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # 设置模型
        self.gf_model = HierarchicalGraphMatchNetwork(
            node_init_dims=config.gf_config.graph_init_dim,
            arguments=config.gf_config,
            device=self.device
        ).to(self.device)

        self.dxf_model = DXFTransformer(
            d_model=config.dxf_config.d_model,
            num_layers=config.dxf_config.num_layers,
            dim_z=config.dxf_config.dim_z,
            nhead=config.dxf_config.nhead,
            dim_feedforward=config.dxf_config.dim_feedforward,
            dropout=config.dxf_config.dropout,
            latent_dropout=config.dxf_config.latent_dropout
        ).to(self.device)

        # 联合损失
        self.joint_loss = JointLoss(config).to(self.device)

        # 使用统一的优化器
        params = list(self.gf_model.parameters()) + list(self.dxf_model.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),  # 使用默认的动量参数
            eps=1e-8
        )

        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.9,
            patience=15,
            verbose=True,
            min_lr=1e-6  # 设置最小学习率
        )

        # 早停
        self.early_stopping = JointEarlyStopping(config)

        # 混合精度训练
        self.scaler = amp.GradScaler()

        # 梯度累积步数
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        # 创建日志目录
        os.makedirs(config.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            config.log_dir,
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

    def _log(self, message):
        """写入日志"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")



    def train_step(self, gf_batch, dxf_batch):
        try:
            # 处理DXF数据
            entity_type, entity_params = dxf_batch
            dxf_data = {
                'entity_type': entity_type.to(self.device),
                'entity_params': entity_params.to(self.device)
            }
            # 处理GF数据
            x1, x2, adj1, adj2 = gf_batch
            feature_p_init = torch.FloatTensor(x1).to(self.device)
            adj_p = torch.FloatTensor(adj1).to(self.device)
            feature_h_init = torch.FloatTensor(x2).to(self.device)
            adj_h = torch.FloatTensor(adj2).to(self.device)

            # 获取原始特征
            with torch.no_grad():
                feature_p0 = self.gf_model(batch_x_p=feature_p_init.clone(), batch_adj_p=adj_p)
                feature_h0 = self.gf_model(batch_x_p=feature_h_init.clone(), batch_adj_p=adj_h)

            # 创建增强视图
            drop_feature_p1 = self.gf_model.drop_feature(feature_p_init.clone(), self.config.drop_feature1)
            drop_feature_p2 = self.gf_model.drop_feature(feature_p_init.clone(), self.config.drop_feature2)
            drop_edge_p1 = adj_p.clone()
            drop_edge_p2 = adj_p.clone()

            for i in range(adj_p.size()[0]):
                drop_edge_p1[i] = self.gf_model.aug_random_edge(adj_p[i].cpu().numpy(), self.config.drop_edge1)
                drop_edge_p2[i] = self.gf_model.aug_random_edge(adj_p[i].cpu().numpy(), self.config.drop_edge2)

            drop_feature_h1 = self.gf_model.drop_feature(feature_h_init.clone(), self.config.drop_feature1)
            drop_feature_h2 = self.gf_model.drop_feature(feature_h_init.clone(), self.config.drop_feature2)
            drop_edge_h1 = adj_h.clone()
            drop_edge_h2 = adj_h.clone()

            for i in range(adj_h.size()[0]):
                drop_edge_h1[i] = self.gf_model.aug_random_edge(adj_h[i].cpu().numpy(), self.config.drop_edge1)
                drop_edge_h2[i] = self.gf_model.aug_random_edge(adj_h[i].cpu().numpy(), self.config.drop_edge2)

            # 使用混合精度训练
            with amp.autocast():
                # GF模型前向传播
                feature_p1 = self.gf_model(batch_x_p=drop_feature_p1, batch_adj_p=drop_edge_p1)
                feature_p2 = self.gf_model(batch_x_p=drop_feature_p2, batch_adj_p=drop_edge_p2)
                feature_h1 = self.gf_model(batch_x_p=drop_feature_h1, batch_adj_p=drop_edge_h1)
                feature_h2 = self.gf_model(batch_x_p=drop_feature_h2, batch_adj_p=drop_edge_h2)

                # 1. 同一图不同视图的信息融合
                feature_p1, feature_p2 = self.gf_model.matching_layer(feature_p1, feature_p2)
                feature_h1, feature_h2 = self.gf_model.matching_layer(feature_h1, feature_h2)

                # 2. 不同图之间的视图信息融合
                feature_p1, _ = self.gf_model.matching_layer(feature_p1, feature_h0)
                feature_p2, _ = self.gf_model.matching_layer(feature_p2, feature_h0)
                feature_h1, _ = self.gf_model.matching_layer(feature_h1, feature_p0)
                feature_h2, _ = self.gf_model.matching_layer(feature_h2, feature_p0)

                # DXF模型前向传播
                dxf_outputs = self.dxf_model(**dxf_data)

                # 计算联合损失
                gf_outputs = {
                    'feature_p1': feature_p1,
                    'feature_p2': feature_p2,
                    'feature_h1': feature_h1,
                    'feature_h2': feature_h2
                }

                joint_loss, individual_losses = self.joint_loss(
                    gf_outputs, dxf_outputs,
                    return_individual=True
                )

                # 仅对backward的损失进行缩放，不影响返回值
                scaled_loss = joint_loss / self.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()

            # 返回原始损失值，不除以gradient_accumulation_steps
            return joint_loss.item(), individual_losses, None
        except Exception as e:
            self._log(f"Error in train_step: {str(e)}")
            self._log(traceback.format_exc())
            return None, None, e

    def train_epoch(self, gf_loader, dxf_loader, epoch):
        self.gf_model.train()
        self.dxf_model.train()

        total_loss = 0
        total_gf_loss = 0
        total_dxf_loss = 0
        num_batches = 0

        # 确保两个loader的长度相同
        if len(gf_loader) != len(dxf_loader):
            min_len = min(len(gf_loader), len(dxf_loader))
            gf_loader = gf_loader[:min_len]
            dxf_loader = list(dxf_loader)[:min_len]

        pbar = tqdm(total=len(gf_loader), desc=f'Epoch {epoch+1}/{self.config.epochs}')

        # 清零梯度
        self.optimizer.zero_grad()

        for batch_idx, (gf_batch, dxf_batch) in enumerate(zip(gf_loader, dxf_loader)):
            try:
                # 训练步骤
                loss, individual_losses, _ = self.train_step(gf_batch, dxf_batch)

                if loss is not None:
                    total_loss += loss
                    total_gf_loss += individual_losses['gf_loss']
                    total_dxf_loss += individual_losses['dxf_loss']
                    num_batches += 1

                    # 梯度累积
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            itertools.chain(self.gf_model.parameters(), self.dxf_model.parameters()),
                            max_norm=1.0
                        )

                        # 优化器步进
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'gf_loss': f'{individual_losses["gf_loss"]:.4f}',
                        'dxf_loss': f'{individual_losses["dxf_loss"]:.4f}'
                    })

                    if num_batches % self.config.log_interval == 0:
                        self._log(
                            f"Epoch {epoch+1}, Batch {num_batches}: "
                            f"Loss={loss:.4f}, "
                            f"GF_Loss={individual_losses['gf_loss']:.4f}, "
                            f"DXF_Loss={individual_losses['dxf_loss']:.4f}"
                        )

            except Exception as e:
                self._log(f"Error in batch: {str(e)}")
                continue

        pbar.close()

        # 处理最后一个不完整的累积批次
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.gf_model.parameters(), self.dxf_model.parameters()),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        avg_gf_loss = total_gf_loss / max(num_batches, 1)
        avg_dxf_loss = total_dxf_loss / max(num_batches, 1)

        # 更新学习率
        self.scheduler.step(avg_loss)

        return avg_loss, avg_gf_loss, avg_dxf_loss

    def validate(self, gf_loader, dxf_loader):
        """验证函数,确保返回所有必要的损失值"""
        self.gf_model.eval()
        self.dxf_model.eval()

        total_loss = 0
        total_gf_loss = 0
        total_dxf_loss = 0
        num_batches = 0

        with torch.no_grad():
            for gf_batch, dxf_batch in zip(gf_loader, dxf_loader):
                try:
                    # 处理GF数据
                    x1, x2, adj1, adj2 = gf_batch
                    feature_p_init = torch.FloatTensor(x1).to(self.device)
                    adj_p = torch.FloatTensor(adj1).to(self.device)
                    feature_h_init = torch.FloatTensor(x2).to(self.device)
                    adj_h = torch.FloatTensor(adj2).to(self.device)

                    # 获取原始特征
                    feature_p0 = self.gf_model(batch_x_p=feature_p_init.clone(), batch_adj_p=adj_p)
                    feature_h0 = self.gf_model(batch_x_p=feature_h_init.clone(), batch_adj_p=adj_h)

                    # 创建增强视图
                    drop_feature_p1 = self.gf_model.drop_feature(feature_p_init.clone(), self.config.drop_feature1)
                    drop_feature_p2 = self.gf_model.drop_feature(feature_p_init.clone(), self.config.drop_feature2)
                    drop_edge_p1 = adj_p.clone()
                    drop_edge_p2 = adj_p.clone()

                    for i in range(adj_p.size()[0]):
                        drop_edge_p1[i] = self.gf_model.aug_random_edge(adj_p[i].cpu().numpy(), self.config.drop_edge1)
                        drop_edge_p2[i] = self.gf_model.aug_random_edge(adj_p[i].cpu().numpy(), self.config.drop_edge2)

                    drop_feature_h1 = self.gf_model.drop_feature(feature_h_init.clone(), self.config.drop_feature1)
                    drop_feature_h2 = self.gf_model.drop_feature(feature_h_init.clone(), self.config.drop_feature2)
                    drop_edge_h1 = adj_h.clone()
                    drop_edge_h2 = adj_h.clone()

                    for i in range(adj_h.size()[0]):
                        drop_edge_h1[i] = self.gf_model.aug_random_edge(adj_h[i].cpu().numpy(), self.config.drop_edge1)
                        drop_edge_h2[i] = self.gf_model.aug_random_edge(adj_h[i].cpu().numpy(), self.config.drop_edge2)

                    # GF模型前向传播
                    feature_p1 = self.gf_model(batch_x_p=drop_feature_p1, batch_adj_p=drop_edge_p1)
                    feature_p2 = self.gf_model(batch_x_p=drop_feature_p2, batch_adj_p=drop_edge_p2)
                    feature_h1 = self.gf_model(batch_x_p=drop_feature_h1, batch_adj_p=drop_edge_h1)
                    feature_h2 = self.gf_model(batch_x_p=drop_feature_h2, batch_adj_p=drop_edge_h2)

                    # 1. 同一图不同视图的信息融合
                    feature_p1, feature_p2 = self.gf_model.matching_layer(feature_p1, feature_p2)
                    feature_h1, feature_h2 = self.gf_model.matching_layer(feature_h1, feature_h2)

                    # 2. 不同图之间的视图信息融合
                    feature_p1, _ = self.gf_model.matching_layer(feature_p1, feature_h0)
                    feature_p2, _ = self.gf_model.matching_layer(feature_p2, feature_h0)
                    feature_h1, _ = self.gf_model.matching_layer(feature_h1, feature_p0)
                    feature_h2, _ = self.gf_model.matching_layer(feature_h2, feature_p0)

                    gf_outputs = {
                        'feature_p1': feature_p1,
                        'feature_p2': feature_p2,
                        'feature_h1': feature_h1,
                        'feature_h2': feature_h2
                    }

                    # 处理DXF数据
                    entity_type, entity_params = dxf_batch
                    dxf_data = {
                        'entity_type': entity_type.to(self.device),
                        'entity_params': entity_params.to(self.device)
                    }

                    # DXF模型前向传播
                    dxf_outputs = self.dxf_model(**dxf_data)

                    # 计算联合损失
                    loss, individual_losses = self.joint_loss(
                        gf_outputs,
                        dxf_outputs,
                        return_individual=True
                    )

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        total_gf_loss += individual_losses['gf_loss']
                        total_dxf_loss += individual_losses['dxf_loss']
                        num_batches += 1

                except Exception as e:
                    self._log(f"Error in validation: {str(e)}")
                    continue

        # 计算平均损失
        avg_loss = total_loss / max(num_batches, 1)
        avg_gf_loss = total_gf_loss / max(num_batches, 1)
        avg_dxf_loss = total_dxf_loss / max(num_batches, 1)

        return avg_loss, avg_gf_loss, avg_dxf_loss

    def _prepare_gf_batch(self, batch):
        """准备GF模型的批次数据"""
        return {
            'batch_x_p': batch[0].to(self.device),
            'batch_adj_p': batch[1].to(self.device)
        }

    def _prepare_dxf_batch(self, batch):
        """准备DeepDXF模型的批次数据"""
        return {
            'entity_type': batch[0].to(self.device),
            'entity_params': batch[1].to(self.device)
        }

    def train_fold(self, train_loaders, val_loaders, fold_idx):
        """训练一个fold"""
        (gf_train_loader, gf_val_loader), (dxf_train_loader, dxf_val_loader) = train_loaders, val_loaders

        best_val_loss = float('inf')
        for epoch in range(self.config.epochs):
            train_loss, train_gf_loss, train_dxf_loss = self.train_epoch(
                gf_train_loader, dxf_train_loader, epoch
            )

            val_loss, val_gf_loss, val_dxf_loss = self.validate(
                gf_val_loader, dxf_val_loader
            )

            # 记录日志
            self._log(
                f"\nEpoch {epoch+1} Summary (Fold {fold_idx+1}):\n"
                f"Train Loss: {train_loss:.4f} "
                f"(GF: {train_gf_loss:.4f}, DXF: {train_dxf_loss:.4f})\n"
                f"Val Loss: {val_loss:.4f} "
                f"(GF: {val_gf_loss:.4f}, DXF: {val_dxf_loss:.4f})"
            )

            # 基于组合损失的早停检查
            individual_losses = {
                'gf_loss': val_gf_loss,
                'dxf_loss': val_dxf_loss
            }
            models = {'gf': self.gf_model, 'dxf': self.dxf_model}
            if self.early_stopping(val_loss, individual_losses, models, self.optimizer, epoch, fold_idx):
                self._log("Early stopping triggered")
                break

            best_val_loss = min(best_val_loss, val_loss)

        return best_val_loss

    def test(self, test_loaders):
        """测试"""
        gf_test_loader, dxf_test_loader = test_loaders
        test_loss, test_gf_loss, test_dxf_loss = self.validate(
            gf_test_loader, dxf_test_loader
        )

        self._log(
            f"\nTest Results:\n"
            f"Test Loss: {test_loss:.4f} "
            f"(GF: {test_gf_loss:.4f}, DXF: {test_dxf_loss:.4f})"
        )

        return test_loss, test_gf_loss, test_dxf_loss