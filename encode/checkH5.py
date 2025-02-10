#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
checkH5.py

用法示例：
  python checkH5.py --h5_file /path/to/some.h5

脚本功能：
  1. 打开指定的 H5 文件。
  2. 遍历其中的所有数据集 (dataset)，打印其名称、shape、dtype。
  3. 额外打印 'src' 和 'n_num' 等关键字段的内容，方便快速验证。

可以根据需要自行定制，例如只查看某些数据集的部分内容等。
"""

import argparse
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Check contents of an H5 file.")
    parser.add_argument('--h5_file', type=str,default="/home/vllm/encode/data/Geom/TRAIN_4096/011L185090DCE_3.h5",
                        help="Path to the H5 file to be inspected.")
    args = parser.parse_args()
    h5_path = args.h5_file

    print(f"[INFO] Checking H5 file: {h5_path}\n")

    with h5py.File(h5_path, 'r') as f:
        # 打印 H5 文件中的所有键（数据集名字）
        keys = list(f.keys())
        print(f"Datasets in file: {keys}\n")

        for k in keys:
            data = f[k]
            # data[...] or data[()] 都可用于读取内容，视其是标量或数组
            # 先打印 shape / dtype
            if hasattr(data, 'shape'):
                print(f"[DATASET] '{k}': shape={data.shape}, dtype={data.dtype}")
            else:
                print(f"[DATASET] '{k}' (no shape, may be scalar/string)")

            # 若想看部分内容，可考虑只取前几个元素或前几行
            # 例如查看 src 是否为字符串或 bytes:
            if k == 'src':
                value = data[()]  # 读取全部内容（可能是标量或者字符串）
                # 如果可能是 bytes，需要 decode
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                print(f"  - src content: {value}")

            # 如果是标量，比如 n_num
            elif k == 'n_num':
                value = data[()]
                print(f"  - n_num: {value}")

            # 如果想查看 feature_matrix 等数组的具体数值，可小规模打印：
            elif k == 'feature_matrix':
                arr = data[()]
                print(f"  - feature_matrix sample[0,:]: {arr[0,:]}")  # 仅打印第一行特征
                # 若数组很大，一次性打印全部会很耗时，也可只看形状/部分内容

            elif k == 'adj_padded':
                arr = data[()]
                print(f"  - adj_padded shape={arr.shape}, example top-left 5x5:\n {arr[:5,:5]}")

            elif k == 'mask':
                arr = data[()]
                print(f"  - mask sample[:20]: {arr[:20]}")

            elif k == 'pos2d_matrix':
                arr = data[()]
                print(f"  - pos2d_matrix shape={arr.shape}, sample[0,:]: {arr[0,:]}")

            print("")

    print("\n[INFO] Done checking.")

if __name__ == "__main__":
    main()
