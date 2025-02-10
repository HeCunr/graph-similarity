import os
import glob

def count_h5_files(directory):
    # 使用 glob 匹配所有的 .h5 文件
    h5_files = glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)

    # 返回匹配到的文件数量
    return len(h5_files)

if __name__ == "__main__":
    # 指定要搜索的目录
    directory ='/home/vllm/encode/data/Seq/synData_TRAIN_2048'

    if os.path.isdir(directory):
        h5_count = count_h5_files(directory)
        print(f"在目录 '{directory}' 下找到 {h5_count} 个 .h5文件。")
    else:
        print(f"错误：'{directory}' 不是一个有效的目录路径。")# !/user/bin/env python3
# -*- coding: utf-8 -*-
