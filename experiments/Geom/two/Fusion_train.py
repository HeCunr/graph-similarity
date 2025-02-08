# Fusion_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np

from config.Fusion_config import load_fusion_args, create_scheduler
from utils.Fusion_early_stopping import FusionEarlyStopping
from dataset.Geom_dataset import GeomDataset
from utils.Geom_utils import set_seed, generate_pairs_from_batch, prepare_batch_data
from model.GeomLayers.DenseGeomMatching import GraphMatchNetwork
from model.GeomLayers.NodeAggregator import MultiLevelNodeAggregator

from dataset.Seq_dataset import load_h5_files
from torch.utils.data import DataLoader, random_split

from dataset.Fusion_dataset import FusionDataset, fusion_collate_fn

from model.FusionLoss import GeomSeqClipLoss
# 这里用你自己的 compute_geom_contrastive_loss、compute_seq_contrastive_loss、
# 以及 forward_geom_for_fusion, forward_seq_for_fusion
# ----------------------------------------------------------
# --------------- 1) 构建 DataLoader ------------------------
# ----------------------------------------------------------

def build_geom_dataloaders(args):
    """
    返回 geom_train_loader, geom_val_loader, geom_test_loader，均为普通 DataLoader。
    内部做“跨图 pairs”+“增广”在 compute_geom_contrastive_loss 时再做。
    """
    dataset = GeomDataset(data_dir=args.geom_data_dir, args=args)

    # 把 train/val/test 图对象分开
    train_graphs = dataset.get_train_data()
    val_graphs   = dataset.get_val_data()
    test_graphs  = dataset.get_test_data()

    # 将图 list 拆成 batches
    def collate_geom(batch):
        # batch 就是一组 GraphData(因为我们不在此处拼 pairs)
        return batch

    # 这里只是把图对象打包成 DataLoader，每次 yield 一个“batch_size数量的GraphData”列表
    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size_geom,
        shuffle=True, collate_fn=collate_geom, drop_last=False
    )
    val_loader = DataLoader(
        val_graphs, batch_size=args.batch_size_geom,
        shuffle=False, collate_fn=collate_geom, drop_last=False
    )
    test_loader = DataLoader(
        test_graphs, batch_size=args.batch_size_geom,
        shuffle=False, collate_fn=collate_geom, drop_last=False
    )
    return train_loader, val_loader, test_loader


def build_seq_dataloaders(args):
    """
    返回 seq_train_loader, seq_val_loader, seq_test_loader，均为普通 DataLoader。
    """
    seq_concat = load_h5_files(args.seq_data_dir)
    total_size = len(seq_concat)
    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(seq_concat, [train_size,val_size,test_size],
                                             generator=torch.Generator().manual_seed(args.seed)
                                             )

    def collate_seq(batch):
        # batch 里是 [(entity_type, entity_params), ...]
        etypes = []
        eparams= []
        for (t, p) in batch:
            etypes.append(t)
            eparams.append(p)
        # => [B,4096], [B,4096,43]
        return torch.stack(etypes, dim=0), torch.stack(eparams, dim=0)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size_seq,
        shuffle=True, collate_fn=collate_seq, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size_seq,
        shuffle=False, collate_fn=collate_seq, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size_seq,
        shuffle=False, collate_fn=collate_seq, drop_last=False
    )
    return train_loader, val_loader, test_loader


def build_fusion_dataloaders(args):
    """
    返回 fusion_train_loader, fusion_val_loader, fusion_test_loader
    这里把 FusionDataset 也拆分出 train/val/test。
    """
    # 先加载全部
    fusion_dataset = FusionDataset(
        geom_dir=args.geom_data_dir,
        seq_dir=args.seq_data_dir
    )
    total_size = len(fusion_dataset)
    train_size = int(0.7*total_size)
    val_size   = int(0.15*total_size)
    test_size  = total_size - train_size - val_size

    # 拆分
    train_ds, val_ds, test_ds = random_split(fusion_dataset, [train_size,val_size,test_size],
                                             generator=torch.Generator().manual_seed(args.seed)
                                             )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size_fusion,
                              shuffle=True, collate_fn=fusion_collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size_fusion,
                              shuffle=False, collate_fn=fusion_collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size_fusion,
                              shuffle=False, collate_fn=fusion_collate_fn)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------
# --------------- 2) 核心对比损失函数 ------------------------
# ----------------------------------------------------------
import torch.nn.functional as F

def compute_geom_contrastive_loss(graph_batch, geom_model, geom_agg, device, args):
    """
    与你在 Geom_train.py `_train_one_epoch` 中类似：对 batch 内图先两份增广(4组) ->
    跨图 + 跨视图 Matching -> InfoNCE。
    注：graph_batch是GraphData列表(List[GraphData])。在collate里没变成pairs，
    所以这里需要临时 pairs = generate_pairs_from_batch(graph_batch)，再 prepare_batch_data(...).
    """
    if len(graph_batch) < 2:
        # batch里只有1张图，无法成对 => loss=0
        return torch.tensor(0., device=device, requires_grad=True)

    pairs = generate_pairs_from_batch(graph_batch)
    if not pairs:
        return torch.tensor(0., device=device, requires_grad=True)

    features1, adj1, masks1, features2, adj2, masks2 = prepare_batch_data(pairs, device)

    # ----- 以下，与 GeomTrainer._train_one_epoch() 类似 -----
    # 1) 数据增强
    drop_feature1 = geom_model.drop_feature(features1, args.drop_feature1)
    drop_edge1 = torch.stack([
        geom_model.aug_random_edge(a.cpu().numpy(), args.drop_edge1)
        for a in adj1
    ]).to(device)

    drop_feature2 = geom_model.drop_feature(features2, args.drop_feature2)
    drop_edge2 = torch.stack([
        geom_model.aug_random_edge(a.cpu().numpy(), args.drop_edge2)
        for a in adj2
    ]).to(device)

    # 2) 聚合
    pfeat1, padj1, pmask1 = geom_agg(drop_feature1, drop_edge1, masks1)
    pfeat2, padj2, pmask2 = geom_agg(drop_feature2, drop_edge2, masks2)

    # 3) GNN
    z1_view1 = geom_model(pfeat1, padj1, pmask1)
    z1_view2 = geom_model(pfeat1, padj2, pmask1)
    z2_view1 = geom_model(pfeat2, padj1, pmask2)
    z2_view2 = geom_model(pfeat2, padj2, pmask2)

    # 原视图(不drop) => 跨图
    pfeat1_orig, padj1_orig, pmask1_orig = geom_agg(features1.float(), adj1.float(), masks1.float())
    z1_orig = geom_model(pfeat1_orig, padj1_orig, pmask1_orig)

    pfeat2_orig, padj2_orig, pmask2_orig = geom_agg(features2.float(), adj2.float(), masks2.float())
    z2_orig = geom_model(pfeat2_orig, padj2_orig, pmask2_orig)

    # 4) matching_layer
    z1_view1, z1_view2 = geom_model.matching_layer(z1_view1, z1_view2)
    z2_view1, z2_view2 = geom_model.matching_layer(z2_view1, z2_view2)

    # 交叉匹配
    z1_view1, _ = geom_model.matching_layer(z1_view1, z2_orig)
    z1_view2, _ = geom_model.matching_layer(z1_view2, z2_orig)
    z2_view1, _ = geom_model.matching_layer(z2_view1, z1_orig)
    z2_view2, _ = geom_model.matching_layer(z2_view2, z1_orig)

    # 归一化
    z1_view1 = F.normalize(z1_view1, dim=-1)
    z1_view2 = F.normalize(z1_view2, dim=-1)
    z2_view1 = F.normalize(z2_view1, dim=-1)
    z2_view2 = F.normalize(z2_view2, dim=-1)

    # 5) InfoNCE
    loss1 = geom_model.loss(z1_view1, z1_view2)
    loss2 = geom_model.loss(z2_view1, z2_view2)
    loss = 0.5*(loss1 + loss2)
    return loss


def compute_seq_contrastive_loss(seq_batch, seq_model, device, args):
    """
    与 Seq_train.py 相同: 两份增广 => forward => InfoNCE
    seq_batch = (entity_type, entity_params).
    """
    if not hasattr(args, 'loss_weights'):
        args.loss_weights = {'loss_cl_weight': 1.0}
    if seq_batch[0].size(0) < 1:
        return torch.tensor(0., device=device, requires_grad=True)

    (entity_type, entity_params) = seq_batch
    entity_type = entity_type.to(device)
    entity_params= entity_params.to(device)

    # 两次增广
    from utils.Seq_augment import augment_seq_sample
    B = entity_type.size(0)
    if B==0:
        return torch.tensor(0., device=device, requires_grad=True)

    # 直接写成2 augment
    # ...
    # 参考你 Seq_train 逻辑
    # 这里直接复用 your two_augmentations_for_batch:
    from model.SeqLayers.seq_transformer_encoder import SeqTransformer
    from model.SeqLayers.Seq_loss import SeqContrastiveLoss
    from utils.Seq_augment import augment_seq_sample

    # 具体请看回答中写好的two_augmentations_for_batch()
    # 这里为简化，直接写:
    B = entity_type.size(0)
    if B<1:
        return torch.tensor(0., device=device, requires_grad=True)

    # 方式：two_augmentations_for_batch
    # (完整见前述)
    def two_augmentations_for_batch(entity_type_t, entity_params_t):
        B_ = entity_type_t.size(0)
        entity_type_np = entity_type_t.cpu().numpy().astype(np.int32)
        entity_params_np= entity_params_t.cpu().numpy().astype(np.int32)
        aug1t_list=[]
        aug1p_list=[]
        aug2t_list=[]
        aug2p_list=[]
        for i in range(B_):
            arr = np.zeros((4096,44), dtype=np.int32)
            arr[:,0] = entity_type_np[i,:]
            arr[:,1:] = entity_params_np[i,:,:]
            a1 = augment_seq_sample(arr)
            a2 = augment_seq_sample(arr)
            aug1t_list.append(a1[:,0])
            aug1p_list.append(a1[:,1:])
            aug2t_list.append(a2[:,0])
            aug2p_list.append(a2[:,1:])
        aug1_type = torch.from_numpy(np.stack(aug1t_list,0)).long().to(device)
        aug1_param= torch.from_numpy(np.stack(aug1p_list,0)).long().to(device)
        aug2_type = torch.from_numpy(np.stack(aug2t_list,0)).long().to(device)
        aug2_param= torch.from_numpy(np.stack(aug2p_list,0)).long().to(device)
        return aug1_type, aug1_param, aug2_type, aug2_param

    aug1_t, aug1_p, aug2_t, aug2_p = two_augmentations_for_batch(entity_type, entity_params)

    # forward
    proj_z1 = seq_model(aug1_t, aug1_p)  # [B,64,256]
    proj_z2 = seq_model(aug2_t, aug2_p)  # [B,64,256]

    # InfoNCE
    from model.SeqLayers.Seq_loss import SeqContrastiveLoss
    seq_loss_fn = SeqContrastiveLoss(cfg=args, device=device,
                                     batch_size=args.batch_size_seq, temperature=args.tau_seq
                                     )
    outputs = {"proj_z1": proj_z1, "proj_z2": proj_z2}
    loss_dict = seq_loss_fn(outputs)
    return loss_dict["loss_contrastive"]


def forward_geom_for_fusion(batch, geom_model, geom_agg, device):
    """
    与回答中相同: [B,4096,44]-> aggregator-> GNN(collect_intermediate=True)-> get_graph_repr_for_fusion
    """
    geom_feat = batch["geom_feat"].to(device).float()   # [B,4096,44]
    geom_adj  = batch["geom_adj"].to(device).float()    # [B,4096,4096]
    geom_mask = batch["geom_mask"].to(device).float()   # [B,4096]

    x_agg, adj_agg, mask_agg = geom_agg(geom_feat, geom_adj, geom_mask)
    out, all_layers = geom_model(x_agg, adj_agg, mask=mask_agg, collect_intermediate=True)
    graph_repr = geom_model.get_graph_repr_for_fusion(all_layers, mask_agg)  # => [B,256]
    return graph_repr.float()


def forward_seq_for_fusion(fus_batch, seq_model, device):
    """
    输入:
        fus_batch["seq_data"]: [B, 4096, 44]
    输出:
        memory_proj: [B, 64, 256]
        fused_vec: [B, 256]
    """
    seq_data = fus_batch["seq_data"].to(device)  # [B, 4096, 44]

    # 确保seq_data是3维[B, 4096, 44]
    if len(seq_data.shape) == 4:  # 如果是[B, 1, 4096, 44]
        seq_data = seq_data.squeeze(1)  # 移除多余的维度


    # 第0列是entity_type => [B,4096]
    entity_type = seq_data[:, :, 0].long()

    # 第1~43列是entity_params => [B,4096,43]
    entity_params = seq_data[:, :, 1:].long()


    # 调用seq_model时明确指定return_fusion=True
    memory_proj, fused_vec = seq_model(
        entity_type,      # [B,4096]
        entity_params,    # [B,4096,43]
        return_fusion=True
    )

    return memory_proj, fused_vec


# ----------------------------------------------------------
# --------------- 3) 训练/验证/测试逻辑 ---------------------
# ----------------------------------------------------------

def train_fusion_one_epoch(
        geom_loader, seq_loader, fusion_loader,
        geom_model, geom_agg, seq_model, fusion_loss_fn,
        optimizer, device, args
):
    geom_model.train()
    geom_agg.train()
    seq_model.train()

    total_steps = min(len(geom_loader), len(seq_loader), len(fusion_loader))
    total_loss = 0.0

    # 分别从三个loader里同时取batch
    geo_iter   = iter(geom_loader)
    seq_iter   = iter(seq_loader)
    fus_iter   = iter(fusion_loader)

    for _ in range(total_steps):
        graph_batch = next(geo_iter)       # List[GraphData], batch_size_geom
        seq_batch   = next(seq_iter)       # (entity_type, entity_params)
        fus_batch   = next(fus_iter)       # dict with geom_feat, seq_data, etc.

        # 1) L_GG
        L_GG = compute_geom_contrastive_loss(graph_batch, geom_model, geom_agg, device, args)

        # 2) L_SS
        L_SS = compute_seq_contrastive_loss(seq_batch, seq_model, device, args)

        # 3) L_GS
        geom_vec = forward_geom_for_fusion(fus_batch, geom_model, geom_agg, device)  # [B, 256]
        memory_proj, seq_vec = forward_seq_for_fusion(fus_batch, seq_model, device)
        # 确保 geom_vec 和 seq_vec 都是 [B, 256] 且已经 L2 normalized
        L_GS = fusion_loss_fn(geom_vec, seq_vec)

        # 合并
        loss = args.lambda1*L_GG + args.lambda2*L_SS + args.lambda3*L_GS

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / total_steps


@torch.no_grad()
def validate_fusion(
        geom_loader, seq_loader, fusion_loader,
        geom_model, geom_agg, seq_model, fusion_loss_fn,
        device, args
):
    """验证逻辑：与训练相同，只是不做反向传播"""
    geom_model.eval()
    geom_agg.eval()
    seq_model.eval()

    total_steps = min(len(geom_loader), len(seq_loader), len(fusion_loader))
    total_loss = 0.0

    geo_iter = iter(geom_loader)
    seq_iter = iter(seq_loader)
    fus_iter = iter(fusion_loader)

    for _ in range(total_steps):
        graph_batch = next(geo_iter)
        seq_batch   = next(seq_iter)
        fus_batch   = next(fus_iter)

        # L_GG
        L_GG = compute_geom_contrastive_loss(graph_batch, geom_model, geom_agg, device, args)
        # L_SS
        L_SS = compute_seq_contrastive_loss(seq_batch, seq_model, device, args)
        # L_GS
        geom_vec = forward_geom_for_fusion(fus_batch, geom_model, geom_agg, device)
        # 这里同样要解包
        _, seq_vec = forward_seq_for_fusion(fus_batch, seq_model, device)
        L_GS = fusion_loss_fn(geom_vec, seq_vec)

        total = args.lambda1*L_GG + args.lambda2*L_SS + args.lambda3*L_GS
        total_loss += total.item()

    return total_loss / total_steps


@torch.no_grad()
def test_fusion(
        geom_loader, seq_loader, fusion_loader,
        geom_model, geom_agg, seq_model, fusion_loss_fn,
        device, args
):
    """测试逻辑：同 validate_fusion"""
    return validate_fusion(
        geom_loader, seq_loader, fusion_loader,
        geom_model, geom_agg, seq_model, fusion_loss_fn,
        device, args
    )


# ----------------------------------------------------------
# --------------- 4) 主入口 main() --------------------------
# ----------------------------------------------------------

def main():
    args = load_fusion_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")

    if not args.disable_wandb:
        wandb.init(project="Fusion", name="Fusion_Train", config=vars(args))

    # 1) 构建 DataLoader
    geom_train_loader, geom_val_loader, geom_test_loader = build_geom_dataloaders(args)
    seq_train_loader, seq_val_loader, seq_test_loader    = build_seq_dataloaders(args)
    fusion_train_loader, fusion_val_loader, fusion_test_loader = build_fusion_dataloaders(args)

    # 2) 构建模型
    from model.GeomLayers.DenseGeomMatching import GraphMatchNetwork
    from model.GeomLayers.NodeAggregator import MultiLevelNodeAggregator
    geom_model = GraphMatchNetwork(args.graph_init_dim, args).to(device)
    geom_agg   = MultiLevelNodeAggregator(args.graph_init_dim).to(device)

    from model.SeqLayers.seq_transformer_encoder import SeqTransformer
    seq_model  = SeqTransformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout_seq,
        latent_dropout=0.1,
        use_selfatt_pool=True
    ).to(device)

    # 3) 优化器 + scheduler + earlystop
    all_params = list(geom_model.parameters()) + list(geom_agg.parameters()) + list(seq_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr_init)
    scheduler = create_scheduler(optimizer, args)
    stopper   = FusionEarlyStopping(patience=args.patience, verbose=True,
                                    path="checkpoints/fusion_best.pt")

    # 融合对比
    fusion_loss_fn = GeomSeqClipLoss(args.temperature_fusion)

    best_val_loss = float('inf')

    # 4) 训练循环
    for epoch in range(args.epochs):
        # ---- train ----
        train_loss = train_fusion_one_epoch(
            geom_train_loader,
            seq_train_loader,
            fusion_train_loader,
            geom_model, geom_agg, seq_model, fusion_loss_fn,
            optimizer, device, args
        )
        scheduler.step()

        # ---- val ----
        val_loss = validate_fusion(
            geom_val_loader,
            seq_val_loader,
            fusion_val_loader,
            geom_model, geom_agg, seq_model, fusion_loss_fn,
            device, args
        )

        print(f"[Epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if not args.disable_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0]
            })

        # early stop
        stop = stopper(val_loss, geom_model, seq_model, epoch, optimizer)
        if stop:
            print("Early stopping triggered!")
            break

    # === 5) 加载best, 测试 ===
    stopper.load_checkpoint(geom_model, seq_model, optimizer=None)
    test_loss = test_fusion(
        geom_test_loader,
        seq_test_loader,
        fusion_test_loader,
        geom_model, geom_agg, seq_model, fusion_loss_fn,
        device, args
    )
    print(f"Test Loss: {test_loss:.4f}")

    if not args.disable_wandb:
        wandb.log({"test_loss": test_loss})
        wandb.finish()

    print("Done.")


if __name__=="__main__":
    main()
