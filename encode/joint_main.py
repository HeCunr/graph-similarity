# joint_main.py
import torch
import os
from datetime import datetime
from config.joint_config import get_joint_args, merge_configs
from model.joint_dataset import JointDataLoader
from joint_train import JointTrainer
import numpy as np

def main():
    # 获取配置
    args = get_joint_args()
    config = merge_configs(args)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_index)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(config.log_dir, f'training_{timestamp}.log')
    os.makedirs(config.log_dir, exist_ok=True)

    # 初始化数据加载器
    data_loader = JointDataLoader(config)
    data_stats = data_loader.get_data_stats()

    # 打印训练信息
    print("\nTraining Configuration:")
    print(f"Device: {device}")
    print(f"Log file: {log_path}")
    print("\nDataset Statistics:")
    for k, v in data_stats.items():
        print(f"{k}: {v}")

    # 初始化训练器
    trainer = JointTrainer(config)

    # K折交叉验证
    fold_val_losses = []
    for fold in range(config.n_folds):
        print(f"\nTraining Fold {fold+1}/{config.n_folds}")

        # 获取当前fold的数据加载器
        train_loaders, val_loaders = data_loader.get_fold_loaders(fold)

        # 训练当前fold
        val_loss = trainer.train_fold(train_loaders, val_loaders, fold)
        fold_val_losses.append(val_loss)

        print(f"Fold {fold+1} Validation Loss: {val_loss:.4f}")

    # 输出交叉验证结果
    mean_val_loss = np.mean(fold_val_losses)
    std_val_loss = np.std(fold_val_losses)
    print("\nCross-validation Results:")
    print(f"Mean Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")

    # 测试集评估
    test_loaders = data_loader.get_test_loaders()
    test_loss, test_gf_loss, test_dxf_loss = trainer.test(test_loaders)

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"GF Loss: {test_gf_loss:.4f}")
    print(f"DXF Loss: {test_dxf_loss:.4f}")

if __name__ == '__main__':
    main()