# Fusion_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils.Fusion_early_stopping import FusionEarlyStopping

from config.Fusion_config import load_fusion_args, create_scheduler
from dataset.Geom_dataset import GeomDataset
from dataset.Seq_dataset import SeqDataset
from dataset.Fusion_dataset import get_fusion_dataloader
from model.GeomLayers.DenseGeomMatching import GraphMatchNetwork
from model.GeomLayers.NodeAggregator import MultiLevelNodeAggregator
from model.GeomLayers.JKPooling import JKPooling
from model.FusionLoss import GeomSeqClipLoss
from model.SeqModel import SeqTransformEncoder  # 示例：您自己的 Transformer
# ... 省略若干 import

def main():
    args = load_fusion_args()

    # 1) 初始化随机种子、日志、wandb等 (与之前类似)
    device = torch.device(f"cuda:{args.gpu_index}") if torch.cuda.is_available() else torch.device("cpu")

    # 2) 构建 Geom 数据加载器 (带增强) - 可以复用您在 Geom_train.py 里的 DataLoader 逻辑
    geom_dataset = GeomDataset(args.geom_data_dir, args)
    geom_train_loader = build_geom_dataloader(geom_dataset.get_train_data(), args.batch_size_geom, shuffle=True)
    # 同理 val/test loader...

    # 3) 构建 Seq 数据加载器 (带增强) - 可以复用您在 Seq_train.py 里的 DataLoader
    seq_dataset = SeqDataset(args.seq_data_dir, args)
    seq_train_loader = build_seq_dataloader(seq_dataset.get_train_data(), args.batch_size_seq, shuffle=True)
    # 同理 val/test...

    # 4) 构建 Fusion 数据加载器 (不做增广) - 用于 L_GS
    fusion_loader = get_fusion_dataloader(args.geom_data_dir, args.seq_data_dir,
                                          batch_size=args.batch_size_fusion, shuffle=True)

    # 5) 初始化模型
    # ---- 几何侧 ----
    geom_model = GraphMatchNetwork(node_init_dims=args.graph_init_dim, args=args).to(device)
    geom_pool  = MultiLevelNodeAggregator(in_features=args.graph_init_dim).to(device)
    # 或者, 如果要收集多层输出再 JKPooling:
    # jk_pool = JKPooling(layer_dims=[int(f) for f in args.filters.split('_')], mode='concat').to(device)

    # ---- 序列侧 ----
    seq_model = SeqTransformEncoder(d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, ...).to(device)

    # ---- 融合对比损失 ----
    fusion_loss_fn = GeomSeqClipLoss(temperature=args.temperature_fusion)

    # 优化器 & 调度器
    all_params = list(geom_model.parameters()) + list(geom_pool.parameters()) + list(seq_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr_init)
    scheduler = create_scheduler(optimizer, args)

    # 早停
    early_stopper = FusionEarlyStopping(patience=args.patience,
                                        path="checkpoints/fusion_best.pt",
                                        verbose=True)

    # ----------------- 训练循环 -----------------
    for epoch in range(args.epochs):
        geom_iter    = iter(geom_train_loader)
        seq_iter     = iter(seq_train_loader)
        fusion_iter  = iter(fusion_loader)

        # 取三者中最大的 step 数, 以便每个 epoch 都能把数据“跑均匀”
        steps = max(len(geom_train_loader), len(seq_train_loader), len(fusion_loader))

        total_loss = 0.0

        geom_model.train()
        geom_pool.train()
        seq_model.train()

        for step_i in range(steps):
            # --- 1) 从 geom_iter 取一个 batch => L_GG ---
            try:
                geom_batch = next(geom_iter)
            except StopIteration:
                geom_iter = iter(geom_train_loader)
                geom_batch = next(geom_iter)
            L_GG = compute_geom_contrastive_loss(geom_batch, geom_model, geom_pool, args, device)

            # --- 2) 从 seq_iter 取一个 batch => L_SS ---
            try:
                seq_batch = next(seq_iter)
            except StopIteration:
                seq_iter = iter(seq_train_loader)
                seq_batch = next(seq_iter)
            L_SS = compute_seq_contrastive_loss(seq_batch, seq_model, args, device)

            # --- 3) 从 fusion_iter 取一个 batch => L_GS ---
            try:
                fusion_batch = next(fusion_iter)
            except StopIteration:
                fusion_iter = iter(fusion_loader)
                fusion_batch = next(fusion_iter)

            # fusion_batch 里是 "geom_feat","geom_adj","geom_mask","seq_data"...
            # 先几何侧 forward(无增广)
            geom_feat = fusion_batch['geom_feat'].to(device)    # [B,4096,44]
            geom_adj  = fusion_batch['geom_adj'].to(device)     # [B,4096,4096]
            geom_mask = fusion_batch['geom_mask'].to(device)    # [B,4096]

            with torch.no_grad():
                # 如果不想回传梯度给 geom-geom CL (可选),
                # 否则去掉 no_grad
                pass

            # node_emb = geom_model(geom_feat, geom_adj, mask=geom_mask, collect_intermediate=False)
            # # or your aggregator
            # node_emb_agg, _ = geom_pool(...)  # ...
            # graph_repr_geom = ...
            # graph_repr_geom = F.normalize(your_proj_head(graph_repr_geom))

            graph_repr_geom = get_geom_fusion_repr(geom_feat, geom_adj, geom_mask, geom_model, geom_pool, device)

            # 序列侧
            seq_data = fusion_batch['seq_data'].to(device)      # [B,4096,44]
            seq_graph_repr = get_seq_fusion_repr(seq_data, seq_model, device)
            # seq_graph_repr = F.normalize( selfattpool( transform_encoder(seq_data) ) )

            # 计算 L_GS
            L_GS = fusion_loss_fn(graph_repr_geom, seq_graph_repr)

            # --- 合并总损失 ---
            L_total = args.lambda1 * L_GG + args.lambda2 * L_SS + args.lambda3 * L_GS

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            total_loss += L_total.item()

        # scheduler update
        scheduler.step()

        avg_loss = total_loss / steps
        print(f"[Epoch {epoch+1}] total_loss={avg_loss:.4f}")

        # ---------- 验证 (同理) -----------
        val_loss = validate_fusion(...)  #

        # early stop
        stop = early_stopper(val_loss, geom_model, seq_model, epoch, optimizer)
        if stop:
            print("Early stopping triggered.")
            break

    # 加载最优模型并测试
    # ...

def compute_geom_contrastive_loss(geom_batch, geom_model, geom_pool, args, device):
    """
    您原先在 Geom_train.py -> _train_one_epoch 所做的 L_GG 计算逻辑,
    只是这里简化成一个函数, 返回一个标量 loss.
    """
    return torch.tensor(0.0, requires_grad=True)

def compute_seq_contrastive_loss(seq_batch, seq_model, args, device):
    """
    同理, Seq_train.py 里面的 L_SS 计算逻辑.
    """
    return torch.tensor(0.0, requires_grad=True)

def get_geom_fusion_repr(geom_feat, geom_adj, geom_mask, geom_model, geom_pool, device):
    """
    走一次 "无增广" 的几何前向, 获得 [B,d] 表示, 并 L2 normalize.
    可以类似:
      pooled = geom_pool( geom_feat, geom_adj, geom_mask )
      out = geom_model.final_proj( pooled )
      out = F.normalize(out, dim=-1)
    """
    # 下面仅示例
    x_agg, adj_agg, mask_agg = geom_pool(geom_feat.double(), geom_adj.double(), geom_mask.double())
    # x_agg: [B,128,44], ...
    node_emb = geom_model(x_agg, adj_agg, mask_agg)
    # 这里如果要 JKPooling, 自行 collect all layers...
    # 最后图级pool
    graph_repr = node_emb.mean(dim=1)  # [B, d]
    graph_repr = nn.functional.normalize(graph_repr.float(), dim=-1)
    return graph_repr

def get_seq_fusion_repr(seq_data, seq_model, device):
    """
    同理, 不增广的序列, 走 transformer -> self_att_pool -> proj -> L2
    """
    # (B,4096,44) -> embed -> progressive pool -> transform -> selfattpool -> proj -> L2
    # 以下仅示例
    seq_embed = seq_model(seq_data) # [B, 64, 256] e.g.
    # selfAtt = self_att_pool(seq_embed) # => [B,256]
    # projected = mlp(selfAtt)
    # out = F.normalize(projected, dim=-1)
    out = torch.randn(seq_data.size(0), 256, device=device)  # placeholder
    out = nn.functional.normalize(out, dim=-1)
    return out

if __name__=="__main__":
    main()
