# SimFusion.py
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import json

# ---- 引入我们在 Fusion_train.py 中使用的配置与模型定义 ----
from config.Fusion_config import load_fusion_args
from model.GeomLayers.DenseGeomMatching import GraphMatchNetwork
from model.GeomLayers.NodeAggregator import MultiLevelNodeAggregator
from model.SeqLayers.seq_transformer_encoder import SeqTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Compute similarity between two DXFs using the trained Fusion model.")
    # 模型权重
    parser.add_argument("--checkpoint", type=str,default="/home/vllm/MulConDXF/checkpoints/fusion_best.pt",
                        help="Path to the trained fusion checkpoint (e.g. fusion_best.pt).")

    # DXF1 的输入
    parser.add_argument("--h5_1", type=str, default="/home/vllm/MulConDXF/data/Seq/TEST_4096/QFN28LK(Cu)-90-450 Rev1_5.h5", help="Path to H5 file for DXF1")
    parser.add_argument("--json_1", type=str,default="/home/vllm/MulConDXF/data/Geom/TEST_4096/QFN28LK(Cu)-90-450 Rev1_5.json", help="Path to JSON file for DXF1")

    # DXF2 的输入
    parser.add_argument("--h5_2", type=str,default="/home/vllm/MulConDXF/data/Seq/TEST_4096/QFN22LD(Cu) -532 Rev1_4.h5",  help="Path to H5 file for DXF2")
    parser.add_argument("--json_2", type=str, default="/home/vllm/MulConDXF/data/Geom/TEST_4096/QFN22LD(Cu) -532 Rev1_4.json", help="Path to JSON file for DXF2")

    # 其余可选参数，比如使用哪个 GPU、是否打印更多日志等
    parser.add_argument("--gpu_index", type=str, default='0', help="Which GPU to use, default=0; use -1 for CPU")
    return parser.parse_args()

def load_checkpoint(geom_model, geom_agg, seq_model, ckpt_path, device):
    """
    从 fusion_train.py 训练好的权重文件加载到模型里。
    检查点里通常包含:
      {
        "epoch": ...,
        "geom_state_dict": geom_model.state_dict(),
        "seq_state_dict": seq_model.state_dict(),
        "optimizer_state_dict": ...,
        "val_loss": ...
      }
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    geom_model.load_state_dict(checkpoint["geom_state_dict"])
    seq_model.load_state_dict(checkpoint["seq_state_dict"])
    # aggregator (geom_agg) 通常与 geom_model 一并训练；若没有单独保存也没关系，一般能正常推理
    print(f"[Info] Loaded checkpoint from {ckpt_path}, epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

def parse_geom_json(json_path, max_nodes=4096, feat_dim=44):
    """
    读取单行 JSON，生成:
      geom_feat: [1, max_nodes, feat_dim]
      geom_adj:  [1, max_nodes, max_nodes]
      geom_mask: [1, max_nodes]
    注意: 不做任何数据增强。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        data = json.loads(line)

    n_num = data['n_num']
    features = np.array(data['features'], dtype=np.float32)  # [n_num,44]
    succs = data['succs']

    geom_feat = np.full((max_nodes, feat_dim), -1, dtype=np.float32)
    geom_adj  = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    geom_mask = np.zeros((max_nodes,), dtype=np.float32)

    # 截断到 max_nodes
    actual_n = min(n_num, max_nodes)
    geom_feat[:actual_n, :] = features[:actual_n, :]

    for u in range(actual_n):
        for v in succs[u]:
            if v < actual_n:
                geom_adj[u,v] = 1
                geom_adj[v,u] = 1

    # 自环
    np.fill_diagonal(geom_adj[:actual_n, :actual_n], 1)
    geom_mask[:actual_n] = 1.0

    # 扩展 batch 维度
    geom_feat = geom_feat[np.newaxis, ...]  # => [1, 4096, 44]
    geom_adj  = geom_adj[np.newaxis, ...]   # => [1, 4096, 4096]
    geom_mask = geom_mask[np.newaxis, ...]  # => [1, 4096]
    return geom_feat, geom_adj, geom_mask

def parse_seq_h5(h5_path):
    """
    读取 H5 文件中的 'dxf_vec' => shape [4096,44]，再加上 batch 维度 => [1,4096,44]。
    如果实际读出来是 [1,4096,44]，则仍为两维 + batch 维度共3维；如出现4维(1,1,4096,44)，需 squeeze 掉多余维度。
    """
    with h5py.File(h5_path, 'r') as f:
        dset = f['dxf_vec'][:]  # 可能是 (4096,44) 或 (1,4096,44) 或更复杂
    # 保证最终形状是 [1,4096,44]
    if len(dset.shape) == 2:
        # (4096,44) => (1,4096,44)
        dset = dset[np.newaxis, ...]
    elif len(dset.shape) == 3 and dset.shape[0] > 1:
        # 说明是 (N,4096,44)，只取第0个？
        # 如果本来就只有一个 DXF，常见是 (1,4096,44) => 无需操作
        # 也可能自己需求不同，这里仅示例：取第0个
        dset = dset[0:1,...]
    # 若出现四维 (1,1,4096,44)，也挤掉多余维
    while len(dset.shape) > 3:
        dset = dset.squeeze(0)
    return dset

@torch.no_grad()
def forward_geom_for_inference(geom_feat, geom_adj, geom_mask, geom_model, geom_agg, device):
    """
    做几何分支的推理，返回 [1,256] 向量
    """
    geom_feat_t = torch.from_numpy(geom_feat).to(device)   # [1,4096,44]
    geom_adj_t  = torch.from_numpy(geom_adj).to(device)    # [1,4096,4096]
    geom_mask_t = torch.from_numpy(geom_mask).to(device)   # [1,4096]

    x_agg, adj_agg, mask_agg = geom_agg(geom_feat_t, geom_adj_t, geom_mask_t)
    out, all_layers = geom_model(x_agg, adj_agg, mask=mask_agg, collect_intermediate=True)
    # 池化到 256
    geom_vec = geom_model.get_graph_repr_for_fusion(all_layers, mask_agg)  # => [1,256]
    return geom_vec

@torch.no_grad()
def forward_seq_for_inference(seq_data, seq_model, device):
    """
    做序列分支的推理，返回 [1,256] 向量
    seq_data: numpy array => [1,4096,44]
       第0列 => entity_type => shape [1,4096]
       第1~43列 => entity_params => shape [1,4096,43]
    """
    seq_data_t = torch.from_numpy(seq_data).long().to(device)  # [1,4096,44]
    # 拆分
    entity_type   = seq_data_t[:,:,0]         # => [1,4096]
    entity_params = seq_data_t[:,:,1:]        # => [1,4096,43]

    # forward
    # seq_model(..., return_fusion=True) => 会返回 (memory_proj, fused_vec)
    _, seq_vec = seq_model(entity_type, entity_params, return_fusion=True)
    # seq_vec: [1,256]
    return seq_vec

def main():
    args = parse_args()

    # 1) 读取训练时使用的默认参数（含模型结构），并设置设备
    base_args = load_fusion_args()
    base_args.gpu_index = args.gpu_index

    if torch.cuda.is_available() and int(base_args.gpu_index) >= 0:
        device = torch.device(f"cuda:{base_args.gpu_index}")
    else:
        device = torch.device("cpu")
    print(f"[Info] Using device: {device}")

    # 2) 构建 模型
    geom_model = GraphMatchNetwork(base_args.graph_init_dim, base_args).to(device)
    geom_agg   = MultiLevelNodeAggregator(in_features=base_args.graph_init_dim).to(device)
    seq_model  = SeqTransformer(
        d_model=base_args.d_model,
        num_layers=base_args.num_layers,
        nhead=base_args.nhead,
        dim_feedforward=base_args.dim_feedforward,
        dropout=base_args.dropout_seq,
        latent_dropout=0.1,
        use_selfatt_pool=True
    ).to(device)

    # 3) 加载 Fusion 训练好的权重
    load_checkpoint(geom_model, geom_agg, seq_model, args.checkpoint, device)
    geom_model.eval()
    geom_agg.eval()
    seq_model.eval()

    # 4) 分别加载 DXF1 / DXF2 的几何数据(json) 和 序列数据(h5)
    print(f"[Info] Reading DXF1 from {args.h5_1} & {args.json_1}")
    geom_feat1, geom_adj1, geom_mask1 = parse_geom_json(args.json_1)
    seq_data1 = parse_seq_h5(args.h5_1)

    print(f"[Info] Reading DXF2 from {args.h5_2} & {args.json_2}")
    geom_feat2, geom_adj2, geom_mask2 = parse_geom_json(args.json_2)
    seq_data2 = parse_seq_h5(args.h5_2)

    # 5) 前向推理，分别得到 (geom1, seq1) 和 (geom2, seq2)
    with torch.no_grad():
        geom1 = forward_geom_for_inference(geom_feat1, geom_adj1, geom_mask1, geom_model, geom_agg, device)
        seq1  = forward_seq_for_inference(seq_data1, seq_model, device)

        geom2 = forward_geom_for_inference(geom_feat2, geom_adj2, geom_mask2, geom_model, geom_agg, device)
        seq2  = forward_seq_for_inference(seq_data2, seq_model, device)

        # 将同一DXF的几何向量 + 序列向量合并 => 融合向量
        fused1 = F.normalize((geom1 + seq1) * 0.5, dim=-1)
        fused2 = F.normalize((geom2 + seq2) * 0.5, dim=-1)

        # 计算余弦相似度
        cos_sim = F.cosine_similarity(fused1, fused2, dim=1)  # 标量 => shape [1]
        sim_val = cos_sim.item()   # [-1,1]

        # 映射到 [0,1]
        sim_01 = 0.5 * (sim_val + 1.0)

    print(f"[Result] Cosine similarity in [-1,1] = {sim_val:.4f}")
    print(f"[Result] Similarity in [0,1]       = {sim_01:.4f}")

if __name__ == "__main__":
    main()
