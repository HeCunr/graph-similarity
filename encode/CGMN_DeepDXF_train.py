# !/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from model.CGMN_DeepDXF_model import CGMN_DeepDXF
from dataset.CGMN_dataset import CFGDataset
from dataset.DeepDXF_dataset import DXFDataset
from utils.CGMN_utils import generate_epoch_pair
from utils.DeepDXF_utils import ContrastiveLoss

def train(args):
    # 加载数据集
    cgmn_dataset = CFGDataset(data_dir=args.cgmn_data_dir, batch_size=args.batch_size)
    deepdxf_dataset = DXFDataset(data_dir=args.deepdxf_data_dir)

    cgmn_data_loader = DataLoader(cgmn_dataset, batch_size=args.batch_size, shuffle=True)
    deepdxf_data_loader = DataLoader(deepdxf_dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    model = CGMN_DeepDXF(args).to(args.device)

    # 定义损失函数和优化器
    criterion_cgmn = torch.nn.CosineEmbeddingLoss()
    criterion_deepdxf = ContrastiveLoss(batch_size=args.batch_size, temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for (batch_x_p, batch_adj_p, y), (entity_type, entity_params) in zip(cgmn_data_loader, deepdxf_data_loader):
            batch_x_p, batch_adj_p, y = batch_x_p.to(args.device), batch_adj_p.to(args.device), y.to(args.device)
            entity_type, entity_params = entity_type.to(args.device), entity_params.to(args.device)

            optimizer.zero_grad()

            z, proj_z1, proj_z2 = model(batch_x_p, batch_adj_p, entity_type, entity_params)

            loss_cgmn = criterion_cgmn(proj_z1, proj_z2, y)
            loss_deepdxf = criterion_deepdxf(proj_z1, proj_z2)

            loss = args.cgmn_weight * loss_cgmn + args.deepdxf_weight * loss_deepdxf

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(cgmn_data_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), args.model_path)

if __name__ == "__main__":
    # 定义参数
    args = ArgParser()
    args.cgmn_data_dir = "/path/to/cgmn/data"
    args.deepdxf_data_dir = "/path/to/deepdxf/data"
    args.batch_size = 32
    args.epochs = 100
    args.lr = 0.001
    args.temperature = 0.5
    args.cgmn_weight = 0.5
    args.deepdxf_weight = 0.5
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_path = "CGMN_DeepDXF.pth"

    train(args)