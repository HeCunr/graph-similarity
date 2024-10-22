# !/user/bin/env python3
# -*- coding: utf-8 -*-
log_file = r"/mnt/share/CGMN/CGMN/CFGLogs/OpenSSL_Min10_Max10_InitDims11_Task_classification\BestModels_Repeat_1\OpenSSL_Min10_Max10_InitDims11_Task_classification_Filter_100_100_100_Match_concat_P_100_Agg_lstm_Hidden_100_Epoch_50_Batch_5_lr_0.0001_Dropout_0.1_Global_0_with_agg_max_pool.BestModel"
print(f"Log file: {log_file}")
with open(log_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
