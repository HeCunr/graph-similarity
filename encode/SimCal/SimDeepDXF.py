import torch
import torch.nn.functional as F
import h5py
import argparse
from model.transformer_encoder import DXFTransformer
from model.DeepDXF_dataset import DXFDataset
from torch.utils.data import DataLoader
from config.DeepDXF_config import DXFConfig

class SimDeepDXF:
    def __init__(self, model_path, cfg=None, device=None):
        """
        初始化SimDeepDXF
        Args:
            model_path: 模型权重路径
            cfg: 配置对象
            device: 计算设备
        """
        if cfg is None:
            # 创建默认配置
            args = argparse.Namespace()
            args.batch_size = 1
            args.data_dir = None
            args.learning_rate = 0.001
            args.temperature = 0.07
            args.epochs = 10
            args.loss_type = 'infonce'
            cfg = DXFConfig(args)

        self.cfg = cfg

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # 初始化模型
        self.model = DXFTransformer(
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            dim_z=cfg.dim_z,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            latent_dropout=cfg.latent_dropout
        ).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def load_single_h5(self, h5_path):
        """加载并处理单个h5文件"""
        dataset = DXFDataset(h5_path)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

        all_representations = []

        with torch.no_grad():
            for entity_type, entity_params in dataloader:
                entity_type = entity_type.to(self.device)
                entity_params = entity_params.to(self.device)

                # 获取模型输出，使用representation作为特征向量
                outputs = self.model(entity_type, entity_params)
                representation = outputs["representation"]
                all_representations.append(representation)

        # 如果h5文件中有多个条目，取平均
        final_representation = torch.mean(torch.stack(all_representations), dim=0)
        return final_representation

    def calculate_similarity(self, representation1, representation2, method='cosine'):
        """计算两个特征向量之间的相似度，结果规范到[0, 1]"""
        if method == 'cosine':
            # 标准化向量并计算余弦相似度
            representation1_norm = F.normalize(representation1, p=2, dim=1)
            representation2_norm = F.normalize(representation2, p=2, dim=1)
            similarity = F.cosine_similarity(representation1_norm, representation2_norm)
            # 余弦相似度范围为[-1, 1]，规范到[0, 1]
            similarity = (similarity + 1) / 2
            return similarity.item()

        elif method == 'euclidean':
            # 计算欧氏距离并转换为相似度分数
            distance = torch.cdist(representation1, representation2, p=2)
            # 将距离转换为相似度，范围为[0, 1]
            similarity = 1 / (1 + distance)
            return similarity.item()

        else:
            raise ValueError(f"Unsupported similarity method: {method}")

    def compare_h5_files(self, h5_path1, h5_path2, method='cosine'):
        """比较两个h5文件并返回它们的相似度分数"""
        try:
            # 获取两个文件的特征向量
            representation1 = self.load_single_h5(h5_path1)
            representation2 = self.load_single_h5(h5_path2)

            # 计算并返回相似度
            similarity = self.calculate_similarity(representation1, representation2, method)
            return similarity

        except Exception as e:
            print(f"Error comparing files: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Calculate similarity between DXF files')
    parser.add_argument('--model_path', type=str, default=r"/home/vllm/encode/checkpoints/best_model.pth",
                        help='Path to the trained model checkpoint')
    parser.add_argument('--file1', type=str, default=r"/home/vllm/encode/data/DeepDXF/TEST_4096/QFN19LA(Cu) -502 Rev1_2.h5",
                        help='Path to first h5 file')
    parser.add_argument('--file2', type=str, default=r"/home/vllm/encode/data/DeepDXF/TEST_4096/QFN19LB(Cu) -503 Rev1_2.h5",
                        help='Path to second h5 file')
    parser.add_argument('--method', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='Similarity calculation method')

    args = parser.parse_args()

    # 初始化SimDeepDXF
    sim_dxf = SimDeepDXF(args.model_path)

    # 计算相似度
    similarity = sim_dxf.compare_h5_files(args.file1, args.file2, args.method)

    if similarity is not None:
        print(f"Similarity score: {similarity:.4f}")

if __name__ == "__main__":
    main()