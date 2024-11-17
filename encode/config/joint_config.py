# config/joint_config.py
import os
import argparse
from config.GF_config import args as gf_args
from config.DeepDXF_config import DXFConfig

class JointConfig:
    def __init__(self):
        # 1. 首先加载GF和DXF的配置
        self.gf_config = gf_args

        # 2. 创建DXFConfig参数
        dxf_args = type('Args', (), {
            'data_dir': r'/mnt/share/DeepDXF_CGMN/encode/data/DeepDXF/dxf_vec_4096',
            'batch_size': 5,
            'learning_rate': 0.0001,
            'temperature': 0.5,
            'epochs': 5,
            'loss_type': 'infonce'
        })()
        self.dxf_config = DXFConfig(dxf_args)

        # 3. 然后初始化配置字典
        self._config = {}

        # 4. 更新基础配置参数
        self._config.update({
            'epochs': 50,
            'gf_batch_size': 45,
            'dxf_batch_size': 5,
            'learning_rate': 0.0001,
            'temp': 0.5,

            # Data paths
            'gf_data_dir': r'/mnt/share/DeepDXF_CGMN/encode/data/GF',
            'dxf_data_dir': r'/mnt/share/DeepDXF_CGMN/encode/data/DeepDXF/dxf_vec_4096',

            # Training configuration
            'n_folds': 10,
            'test_size': 0.2,
            'patience': 15,
            'weight_method': 'uncertainty',
            'window_size': 50,
            'gradient_accumulation_steps': 2,
            'scheduler_patience': 15,
            'scheduler_factor': 0.9,

            # 视图增强参数
            'drop_feature1': 0.3,
            'drop_feature2': 0.4,
            'drop_edge1': 0.2,
            'drop_edge2': 0.2,

            # Loss scaling factors
            'gf_loss_scale': 1.0,
            'dxf_loss_scale': 1.0,

            # Optimizer configuration
            'weight_decay': 1e-6,
            'warmup_ratio': 0.1,
            'min_lr': 1e-6,

            # Logging configuration
            'log_dir': os.path.join('logs', 'joint_logs'),
            'model_dir': os.path.join('models', 'joint_models'),
            'log_interval': 10,

            # GPU configuration
            'device': 'cuda',
            'gpu_index': 0,

            # GF模型特有的配置 (现在可以安全地访问self.gf_config了)
            'filters': self.gf_config.filters,
            'conv': self.gf_config.conv,
            'match': self.gf_config.match,
            'perspectives': self.gf_config.perspectives,
            'match_agg': self.gf_config.match_agg,
            'hidden_size': self.gf_config.hidden_size,
            'global_flag': self.gf_config.global_flag,
            'global_agg': self.gf_config.global_agg
        })

    def get(self, key, default=None):
        """Get a configuration value with a default fallback"""
        return self._config.get(key, default)

    def __getattr__(self, name):
        """Allow access to config values as attributes"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'JointConfig' object has no attribute '{name}'")

    def update(self, **kwargs):
        """Update configuration values"""
        self._config.update(kwargs)

def get_joint_args():
    parser = argparse.ArgumentParser(description='Joint Training Config')

    # Data path parameters
    parser.add_argument('--gf_data', type=str, default=None, help='GF model data directory')
    parser.add_argument('--dxf_data', type=str, default=None, help='DXF model data directory')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gf_batch_size', type=int, default=45)
    parser.add_argument('--dxf_batch_size', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_method', type=str, default='uncertainty',
                        choices=['uncertainty', 'grad_norm', 'loss_ratio'])
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.001)

    # GPU parameters
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()

def merge_configs(args):
    """Merge command line arguments with default configuration"""
    config = JointConfig()

    # Update configuration with command line arguments
    updates = {
        'gf_data_dir': args.gf_data or config.get('gf_data_dir'),
        'dxf_data_dir': args.dxf_data or config.get('dxf_data_dir'),
        'epochs': args.epochs,
        'gf_batch_size': args.gf_batch_size,
        'dxf_batch_size': args.dxf_batch_size,
        'patience': args.patience,
        'weight_method': args.weight_method,
        'temp': args.temp,
        'learning_rate': args.lr,
        'device': args.device,
        'gpu_index': args.gpu_index
    }

    config.update(**updates)
    return config

# Default configuration instance
default_config = JointConfig()