# !/user/bin/env python3
# -*- coding: utf-8 -*-
import torch

# 假设 best_model_path 是保存的模型文件的路径
best_model_path ='/mnt/share/CGMN/CGMN/CFGLogs/OpenSSL_Min10_Max10_InitDims11_Task_classification\BestModels_Repeat_1\OpenSSL_Min10_Max10_InitDims11_Task_classification_Filter_100_100_100_Match_concat_P_100_Agg_lstm_Hidden_100_Epoch_50_Batch_5_lr_0.0001_Dropout_0.1_Global_0_with_agg_max_pool.BestModel'

# 加载模型权重
model_state_dict = torch.load(best_model_path)

# 打印模型的所有权重和参数
for key, value in model_state_dict.items():
    print(f"Layer: {key} | Shape: {value.shape}")
    print(value)
