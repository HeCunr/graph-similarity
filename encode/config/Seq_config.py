# config/Seq_config.py
class SeqConfig:
    def __init__(self, args):
        # 默认数据目录
        self.data_dir = args.data_dir or r'/home/vllm/encode/data/Seq/TRAIN_4096'

        # 模型参数
        self.d_model = 256
        self.num_layers = 6
        self.dim_z = 256
        self.nhead = 8
        self.dim_feedforward = 512
        self.dropout = 0.2
        self.latent_dropout = 0.3

        # 训练参数
        self.batch_size = 32
        self.initial_lr = 1e-5  # 初始学习率
        self.max_lr = 1e-4       # 最大学习率
        self.final_lr = 1e-5     # 最终学习率
        self.warmup_epochs = 10  # 学习率预热阶段轮数（占总训练轮数的10%）
        self.epochs = 100        # 总训练轮数
        self.temperature = 0.07  # InfoNCE损失的温度参数
        self.gpu_id = getattr(args, 'gpu_id', 1)  # 默认使用 GPU1

        # 数据集划分比例
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # 梯度裁剪与早停
        self.clip_grad_norm = 1.0
        self.patience = 20       # 早停耐心值
        self.delta = 0.001       # 早停最小改善阈值

        # 损失权重
        self.loss_weights = {'loss_cl_weight': 1.0}

        # 正则化
        self.weight_decay = 1e-4

        # wandb配置
        self.wandb_project = args.wandb_project or "Seq"
        self.wandb_entity = args.wandb_entity or "102201525-fuzhou-university"
        self.wandb_name = args.wandb_name or "Seq"
        self.use_wandb = True