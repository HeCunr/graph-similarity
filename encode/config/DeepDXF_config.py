class DXFConfig:
    def __init__(self, args):
        # 默认数据目录
        self.data_dir = args.data_dir or r'/home/vllm/encode/data/DeepDXF/TRAIN_4096'

        # 模型参数
        self.d_model = 256
        self.num_layers = 4
        self.dim_z = 256
        self.nhead = 8
        self.dim_feedforward = 512
        self.dropout = 0.2
        self.latent_dropout = 0.2
        self.latent_dropout = 0.1

        # 训练参数
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.temperature = max(0.07, args.temperature)  # 确保温度参数不会太小
        self.epochs = args.epochs
        self.cl_loss_type = args.loss_type

        # 添加梯度裁剪
        self.clip_grad_norm = 1.0
        # 早停参数
        self.patience = 7  # 早停耐心值
        self.delta = 0.001  # 最小改善阈值

        # 交叉验证参数
        self.n_folds = 10  # K折交叉验证的折数
        self.test_size = 0.2  # 测试集比例

        # 损失权重
        self.loss_weights = {
            'loss_cl_weight': 1.0,
        }

        # 数据增强参数
        self.use_data_augment = True
        self.max_total_len = 4096

        # 其他参数
        self.save_frequency = 1
        self.warmup_steps = 1000

        # 优化器参数
        self.weight_decay = 1e-4  # 添加权重衰减

        # 添加wandb配置
        self.wandb_project = "DeepDXF"  # wandb项目名称
        self.wandb_entity = "102201525-fuzhou-university"    # 你的wandb用户名
        self.wandb_name = "DeepDXF"          # 本次运行的名称
        self.use_wandb = True           # 是否使用wandb