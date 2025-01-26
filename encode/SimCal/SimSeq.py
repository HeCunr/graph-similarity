#!/usr/bin/env python3
# SimSeq.py
"""
计算两个h5文件之间的相似度，使用已训练好的SeqTransformer权重。
相似度采用余弦相似度，并归一化到 [0,1]。
丢弃投影头与BN，仅使用 embedding->progressive_pool->encoder 提取表征。
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np

# ---- 你训练时的主干模型与数据集加载 ----
from model.SeqLayers.seq_transformer_encoder import SeqTransformer
import h5py

# ========== 1. 定义一个子模型，只保留“embedding->progressive_pool->encoder” ==========

class SeqBackbone(torch.nn.Module):
    """
    从 SeqTransformer 中只保留embedding, progressive_pool, encoder三部分，
    丢弃 projection/mlp, BN, dropout。
    """
    def __init__(self, full_model: SeqTransformer):
        super().__init__()
        self.embedding = full_model.embedding
        self.progressive_pool = full_model.progressive_pool
        self.encoder = full_model.encoder
        # 不再使用 projection, bn, dropout

    @torch.no_grad()
    def forward(self, entity_type, entity_params):
        """
        返回: (batch_size, 64, 256) 的编码序列
             或者也可以在这里做序列平均，得到 (batch_size, 256)
        """
        # 1) 嵌入 => (B, 4096, 256)
        src = self.embedding(entity_type, entity_params)

        # 2) progressive_pool => (B, 64, 256)
        src = self.progressive_pool(src)

        # 3) 变成 [seq_len=64, batch=B, d_model=256] 喂 transformer
        src = src.permute(1, 0, 2)  # => (64, B, 256)
        memory = self.encoder(src)  # => (64, B, 256)
        memory = memory.permute(1, 0, 2)  # => (B, 64, 256)

        return memory

# ========== 2. 从一个 h5 文件提取所有样本的平均表征 ==========

@torch.no_grad()
def extract_file_representation(h5_path, backbone, device="cpu"):
    """
    给定一个 .h5 文件(内含多条 dxf_vec)，
    使用 backbone(embedding->pool->encoder) 提取其所有样本的表征，
    最后做平均 => 得到 (256,) 的文件级向量。
    """
    # 读取 .h5
    with h5py.File(h5_path, 'r') as f:
        if 'dxf_vec' not in f:
            raise ValueError(f"No dataset 'dxf_vec' found in {h5_path}.")
        dset = f['dxf_vec']
        n_samples = len(dset)

    all_vecs = []
    # 每次读一个样本，做 forward
    for i in range(n_samples):
        with h5py.File(h5_path, 'r') as f:
            data_i = f['dxf_vec'][i]  # shape=(4096,44)

        # 拆分 entity_type, entity_params
        entity_type = torch.tensor(data_i[:, 0], dtype=torch.long, device=device).unsqueeze(0)
        entity_params = torch.tensor(data_i[:, 1:], dtype=torch.float, device=device).unsqueeze(0)
        # => (1,4096), (1,4096,43)

        # 前向 => 得到 (1, 64, 256)
        memory = backbone(entity_type, entity_params)  # shape=(1,64,256)

        # 可以对 seq_len=64 做平均，得到 (1,256)
        # 也可以保留序列再处理，这里做简单平均
        memory_mean = memory.mean(dim=1)  # => (1,256)

        all_vecs.append(memory_mean.cpu().numpy())  # => 形如(1,256)

    # 拼起来 => (n_samples, 256)
    all_vecs = np.concatenate(all_vecs, axis=0)  # shape=(n_samples,256)
    # 对 n_samples 再做平均 => (256,)
    file_vector = all_vecs.mean(axis=0)
    return file_vector


# 在原有代码末尾添加以下类（或适当整合到原有逻辑中）

class SimSeq:
    def __init__(self, model_ckpt_path, device="cpu"):
        self.device = device
        # 加载完整模型
        self.full_model = SeqTransformer(
            d_model=256,
            num_layers=6,
            dim_z=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            latent_dropout=0.3
        )
        checkpoint = torch.load(model_ckpt_path, map_location=device)
        self.full_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # 构建backbone
        self.backbone = SeqBackbone(self.full_model).to(device)
        self.backbone.eval()

    @torch.no_grad()
    def extract_file_representation(self, h5_path):
        """直接调用原有的提取函数"""
        return extract_file_representation(h5_path, self.backbone, device=self.device)
# ========== 3. 主函数：加载 best_model, 构建 backbone，计算余弦相似度 ==========

def main():
    parser = argparse.ArgumentParser("SimSeq: compute similarity of two .h5 files by Seq.")
    parser.add_argument("--h5_file1", type=str, default=r"/home/vllm/encode/data/Seq/TEST_4096/QFN28LK(Cu)-90-450 Rev1_2.h5", help="Path to the first h5 file")
    parser.add_argument("--h5_file2", type=str,default=r"/home/vllm/encode/data/Seq/TEST_4096/QFN22LD(Cu) -532 Rev1_1.h5", help="Path to the second h5 file")
    parser.add_argument("--model_ckpt", type=str, default=r"/home/vllm/encode/pretrained/Seq/Seq.pth", help="Trained model checkpoint path")
    parser.add_argument("--gpu_id", type=int, default=1, help="Which GPU to use if available.")
    args = parser.parse_args()

    # 1) 设备选择
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 2) 加载完整模型 + state_dict
    full_model = SeqTransformer(
        d_model=256,
        num_layers=6,
        dim_z=256,
        nhead=8,
        dim_feedforward=512,
        dropout=0.2,
        latent_dropout=0.3
    )
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    full_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # 3) 构建 backbone 只保留 embedding->progressive_pool->encoder
    backbone = SeqBackbone(full_model).to(device)
    backbone.eval()

    # 4) 分别提取两个文件的表征向量
    vecA = extract_file_representation(args.h5_file1, backbone, device=device)  # (256,)
    vecB = extract_file_representation(args.h5_file2, backbone, device=device)  # (256,)

    # 5) 余弦相似度 => [-1,1]，然后归一化到 [0,1]
    #   cos_sim = cos(vecA, vecB) = (A·B)/(||A||*||B||)
    #   norm_sim = (cos_sim + 1) / 2
    a_t = torch.tensor(vecA, dtype=torch.float, device=device)
    b_t = torch.tensor(vecB, dtype=torch.float, device=device)
    # 先归一化
    a_norm = F.normalize(a_t, dim=0)
    b_norm = F.normalize(b_t, dim=0)
    cos_sim = torch.dot(a_norm, b_norm).item()  # scalar in [-1,1]
    sim_01 = (cos_sim + 1.0) / 2.0  # => [0,1]

    # 6) 打印结果
    print(f"Cosine similarity (raw)   = {cos_sim:.4f}")
    print(f"Similarity in [0,1] range= {sim_01:.4f}")

if __name__ == "__main__":
    main()
