# config/GF_config.py
import argparse

parser = argparse.ArgumentParser(description="GF model for graph matching")

# 原有参数保持不变，去除不必要的交叉验证参数即可
# 例如 n_folds、test_split 可以直接删除，也可保留但不使用。
# 这里我们直接删去 `n_folds`, `test_split`，并增加 train_split, val_split, test_split 三个参数

# Data parameters
parser.add_argument('--data_dir', type=str, default='/home/vllm/encode/data/GF/discreted_dataset/TRAIN',
                    help='root directory for the graph dataset')
parser.add_argument('--dataset', type=str, default="PROTEINS",
                    help='name of the dataset')
parser.add_argument('--graph_size_min', type=int, default=2,
                    help='minimum number of nodes in a graph')
parser.add_argument('--graph_size_max', type=int, default=200,
                    help='maximum number of nodes in a graph')
parser.add_argument('--graph_init_dim', type=int, default=6,
                    help='initial feature dimension for graph nodes')

# Model architecture parameters (保持不变)
parser.add_argument("--filters", type=str, default='100_100_100',
                    help="Filter dimensions for graph convolution network")
parser.add_argument("--conv", type=str, default='gcn',
                    help="Type of GNN layer (gcn/graphsage/gin/ggnn)")
parser.add_argument("--match", type=str, default='concat',
                    help="Node matching method")
parser.add_argument("--perspectives", type=int, default=100,
                    help='number of perspectives for matching')
parser.add_argument("--match_agg", type=str, default='lstm',
                    help="Aggregation method for matched features")
parser.add_argument("--hidden_size", type=int, default=100,
                    help='hidden size for graph-level embedding')

# Data augmentation parameters (保持不变)
parser.add_argument('--drop_feature1', type=float, default=0.3,
                    help='feature dropout rate for first view')
parser.add_argument('--drop_feature2', type=float, default=0.4,
                    help='feature dropout rate for second view')
parser.add_argument('--drop_edge1', type=float, default=0.2,
                    help='edge dropout rate for first view')
parser.add_argument('--drop_edge2', type=float, default=0.2,
                    help='edge dropout rate for second view')

# Training parameters
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=128,
                    help='number of graphs per batch')

# 去除 n_folds、test_split 等相关参数，改为手动指定三份数据划分比例
parser.add_argument('--train_split', type=float, default=0.7,
                    help='proportion of data for training')
parser.add_argument('--val_split', type=float, default=0.15,
                    help='proportion of data for validation')
parser.add_argument('--test_split', type=float, default=0.15,
                    help='proportion of data for testing')

parser.add_argument('--patience', type=int, default=30,
                    help='patience for early stopping')
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate")
parser.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout probability")
parser.add_argument("--tau", type=float, default=0.7,
                    help="Temperature parameter for contrastive loss")

# Global information parameters (保持不变)
parser.add_argument("--global_flag", type=lambda x: (str(x).lower() == 'true'),
                    default='false', help="Whether to use global information")
parser.add_argument("--global_agg", type=str, default='max_pool',
                    help="Aggregation function for global level")

# System parameters (保持不变)
parser.add_argument('--gpu_index', type=str, default='1',
                    help="GPU index to use")
parser.add_argument('--log_path', type=str, default='../logs/',
                    help='path for log files')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for reproducibility')

# Wandb parameters (保持不变)
parser.add_argument('--wandb_project', type=str, default="GF",
                    help='Weights & Biases project name')
parser.add_argument('--wandb_entity', type=str, default="102201525-fuzhou-university",
                    help='Weights & Biases entity (username or team name)')
parser.add_argument('--wandb_run_name', type=str, default="GF-Dismantle",
                    help='Weights & Biases run name')
parser.add_argument('--wandb_log_freq', type=int, default=100,
                    help='Frequency of logging model gradients and parameters')
parser.add_argument('--disable_wandb', action='store_true',
                    help='Disable Weights & Biases logging')

# Parse arguments
gf_args = parser.parse_args()
