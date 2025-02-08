# SimSeq.py
import torch
import h5py
import numpy as np
import os
import argparse

# ---- 根据你本地项目的结构作相应导入 ----
from model.SeqLayers.seq_transformer_encoder import SeqTransformer
from model.SeqLayers.Seq_embedding import SeqEmbedding, PositionalEncodingLUT, nn
from config.Seq_config import SeqConfig


class SeqTransformerNoProjection(SeqTransformer):
    """
    继承自训练时使用的 SeqTransformer，但在 forward 中去除投影头网络。
    仅保留嵌入、渐进池化、TransformerEncoder。
    """
    def forward(self, entity_type, entity_params):
        """
        输入:
            entity_type   (B,4096)
            entity_params (B,4096,43)
        输出:
            memory (B,64,256)
        注意：不再执行 self.projection 等投影操作
        """
        # 1) 嵌入
        src = self.embedding(entity_type, entity_params)  # => (B, 4096, 256)

        # 2) Progressive Pooling
        src = self.progressive_pool(src)                  # => (B, 64, 256)

        # 3) Transformer 编码: 先转 [seq_len, batch, d_model]
        src = src.permute(1, 0, 2)                        # => (64, B, 256)
        memory = self.encoder(src)                        # => (64, B, 256)
        memory = memory.permute(1, 0, 2)                  # => (B, 64, 256)

        return memory


def load_single_h5_sample(h5_path):
    """
    加载单个 H5 文件，假设其中只有 1 个样本，数据集名称为 'dxf_vec'。
    返回 entity_type (1,4096) 和 entity_params (1,4096,43) 的 torch.Tensor。
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        if 'dxf_vec' not in f:
            raise KeyError(f"[{h5_path}] does not contain dataset 'dxf_vec'.")
        data = f['dxf_vec'][0]  # 假设只取第 0 条样本 => shape (4096, 44)

    # 拆分 => entity_type (4096,) + entity_params (4096,43)
    entity_type_np = data[:, 0].astype(np.int64)
    entity_params_np = data[:, 1:].astype(np.int64)

    # 增加 batch 维度 (1,4096) / (1,4096,43)
    entity_type_t = torch.from_numpy(entity_type_np).unsqueeze(0)
    entity_params_t = torch.from_numpy(entity_params_np).unsqueeze(0)
    return entity_type_t, entity_params_t


def build_model_and_load_weights(model_path, cfg):
    """
    根据 cfg 构建 SeqTransformerNoProjection 模型，并加载已训练的权重。
    """
    model = SeqTransformerNoProjection(
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        dim_z=cfg.dim_z,
        nhead=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        latent_dropout=cfg.latent_dropout
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')
    # 直接加载训练时的参数
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model


def get_feature_vector(model, entity_type, entity_params):
    """
    通过模型得到 (B,64,256) => 在序列维度做 mean pooling => 得到 (B,256) 的表示。
    假设 B=1，则返回 (256,) 的向量。
    """
    with torch.no_grad():
        memory = model(entity_type, entity_params)  # (B,64,256)
        pooled = memory.mean(dim=1)                # (B,256)
    return pooled.squeeze(0)  # => (256,)


class SimSeq:
    """
    封装一个类，提供模型加载与文件向量提取功能。
    在 Seq_FMRD.py 中可直接:
        from SimSeq import SimSeq
        sim_seq = SimSeq(model_path="xxx", device="cuda:0")
        vec = sim_seq.extract_file_representation("/path/to/some.h5")
    """
    def __init__(self, model_path, device='cpu'):
        self.device = device

        # 1) 构造与训练一致的配置
        self.cfg = SeqConfig(None)
        self.cfg.d_model = 256
        self.cfg.num_layers = 6
        self.cfg.dim_z = 256
        self.cfg.nhead = 8
        self.cfg.dim_feedforward = 512
        self.cfg.dropout = 0.2
        self.cfg.latent_dropout = 0.3

        # 2) 构建模型并加载权重
        self.model = build_model_and_load_weights(model_path, self.cfg)
        self.model.to(self.device)

    def extract_file_representation(self, h5_path):
        """
        给定 h5 文件路径，返回其 (256,) 的向量（numpy 格式），
        若出错则返回 None。
        """
        if not os.path.exists(h5_path):
            print(f"[Warning] File not found: {h5_path}")
            return None

        try:
            entity_type, entity_params = load_single_h5_sample(h5_path)
            entity_type = entity_type.to(self.device)
            entity_params = entity_params.to(self.device)

            vec = get_feature_vector(self.model, entity_type, entity_params)
            return vec.cpu().numpy()
        except Exception as e:
            print(f"[Error] Failed to extract representation for {h5_path}: {e}")
            return None


def main():
    """
    这是原先SimSeq脚本的main()，用来对比两个文件的相似度。
    如无需此脚本式功能，可删除或保留。
    """
    parser = argparse.ArgumentParser(description="Compute similarity between two H5 samples.")
    parser.add_argument("--model_path", type=str, default="/home/vllm/encode/checkpoints/Seq/Seq_align.pth",
                        help="Path to the trained model checkpoint (e.g. best_model.pth)")
    parser.add_argument("--h5_file1", type=str, default="/home/vllm/encode/data/Seq/TEST_4096/QFN28LK(Cu)-90-450 Rev1_3.h5",
                        help="Path to the first h5 file (each contains 1 sample).")
    parser.add_argument("--h5_file2", type=str, default="/home/vllm/encode/data/Seq/TEST_4096/QFN28LK(Cu)-90-450 Rev1_3.h5",
                        help="Path to the second h5 file (each contains 1 sample).")
    args = parser.parse_args()

    sim_seq = SimSeq(args.model_path, device="cpu")

    vec1 = sim_seq.extract_file_representation(args.h5_file1)
    vec2 = sim_seq.extract_file_representation(args.h5_file2)
    if vec1 is None or vec2 is None:
        print("Failed to load vectors.")
        return

    # 余弦相似度 => [0,1] 映射
    v1 = torch.tensor(vec1, dtype=torch.float32)
    v2 = torch.tensor(vec2, dtype=torch.float32)
    sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()
    sim_01 = 0.5*(sim+1.0)

    print(f"Similarity in [0,1]: {sim_01:.4f}")

if __name__ == "__main__":
    main()
