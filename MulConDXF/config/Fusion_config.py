# config/Fusion_config.py
import argparse
import math
import torch

def load_fusion_args():
    parser = argparse.ArgumentParser(description="Fusion config for Geometry-Sequence model")

    # ----------------------
    # 通用参数
    # ----------------------
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch_size_geom", type=int, default=32, help="Batch size for Geom branch")
    parser.add_argument("--batch_size_seq", type=int, default=32, help="Batch size for Seq branch")
    parser.add_argument("--batch_size_fusion", type=int, default=32, help="Batch size for Fusion loader (un-augmented)")

    parser.add_argument("--gpu_index", type=str, default='0', help="Which GPU to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=30, help="Patience for early stopping")

    # ----------------------
    # 数据集路径
    # ----------------------
    # Geom 原始数据目录
    parser.add_argument('--geom_data_dir', type=str, default='/home/vllm/MulConDXF/data/Geom/TRAIN_4096',
                        help='root directory for the geometric graph dataset (.json)')
    # Seq 原始数据目录
    parser.add_argument('--seq_data_dir', type=str, default='/home/vllm/MulConDXF/data/Seq/TRAIN_4096',
                        help='root directory for the sequence dataset (.h5)')

    # ----------------------
    # 几何侧模型相关
    # ----------------------
    parser.add_argument("--graph_size_max", type=int, default=4096,
                        help="max number of nodes in a graph")
    parser.add_argument("--graph_init_dim", type=int, default=44,
                        help="initial node feature dim in Geom")
    parser.add_argument("--filters", type=str, default='100_100_100',
                        help="Filter dimensions for the GNN (e.g. '100_100_100')")
    parser.add_argument("--conv", type=str, default='ggnn', help="Type of GNN (gcn/graphsage/gin/ggnn)")
    parser.add_argument("--match", type=str, default='concat', help="Node matching method")
    parser.add_argument("--perspectives", type=int, default=100, help="Matching perspectives")

    # 数据增强(Geom)
    parser.add_argument('--drop_feature1', type=float, default=0.4)
    parser.add_argument('--drop_feature2', type=float, default=0.1)
    parser.add_argument('--drop_edge1', type=float, default=0.2)
    parser.add_argument('--drop_edge2', type=float, default=0.3)
    parser.add_argument("--tau_geom", type=float, default=0.7, help="Temperature for Geom-Geom CL")
    parser.add_argument("--dropout_geom", type=float, default=0.1)
    # ----------------------
    # 序列侧模型相关
    # ----------------------
    # 示例：transformer 的一些参数
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout_seq", type=float, default=0.1)
    parser.add_argument("--tau_seq", type=float, default=0.7, help="Temperature for Seq-Seq CL")
    # ----------------------
    # 学习率调度 & 优化器
    # ----------------------
    parser.add_argument("--lr_init", type=float, default=1e-5, help="initial LR")
    parser.add_argument("--lr_peak", type=float, default=1e-4, help="LR at the end of warmup")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="final minimal LR in cos decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="fraction of epochs for linear warmup to lr_peak, then cos decay back to min_lr")
    # ----------------------
    # 融合对比学习部分
    # ----------------------
    # 在 load_fusion_args() 里，和其他 parser.add_argument 同级位置，添加：
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='proportion of data for training')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='proportion of data for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='proportion of data for testing')

    parser.add_argument("--temperature_fusion", type=float, default=0.07,
                            help="Temperature for Geom-Seq CL (CLIP style)")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for L_GG")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Weight for L_SS")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Weight for L_GS")
    # ----------------------
    # 其余如 wandb/logging ...
    # ----------------------
    parser.add_argument("--disable_wandb", action='store_true', help="Disable W&B logging")
    parser.add_argument("--log_path", type=str, default="/home/vllm/MulConDXF/logs/Fusion")

    args = parser.parse_args()
    return args


def create_scheduler(optimizer, args):
    """
    按照：
      - 前10% epoch: 线性从 lr_init -> lr_peak
      - 剩余90%: 余弦衰减到 min_lr
    """
    warmup_epochs = int(args.epochs * args.warmup_ratio)
    total_epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性从 lr_init -> lr_peak
            alpha = epoch / float(warmup_epochs)
            return (1 - alpha) * (args.lr_init) + alpha * (args.lr_peak)
        else:
            # 余弦衰减: epoch从 warmup_epochs -> total_epochs
            progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            # 余弦从 lr_peak -> min_lr
            cos_scale = 0.5*(1 + math.cos(math.pi * progress))
            return args.min_lr + cos_scale*(args.lr_peak - args.min_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler
