# dxf_process.py

import os
import sys

# 确保可以导入其他脚本中的函数
# 假设 dxf_process_DeepDXF.py 和 dxf_process_CGMN.py 与 dxf_process.py 位于同一目录
from dxf_process_DeepDXF import process_dxf_files_for_deepdxf
from dxf_process_CGMN import process_dxf_files_for_cgmn

def main():
    # 指定输入 DXF 文件的目录
    input_dir = r'/mnt/share/DeepDXF_CGMN/data/DXF'  # 修改为您的 DXF 文件目录

    # 指定输出 H5 文件的目录
    output_h5_dir = r'/mnt/share/DeepDXF_CGMN/data/DeepDXF/dxf_vec'  # 修改为您希望保存 H5 文件的目录

    # 指定输出 JSON 文件的路径
    output_json_file = r'/mnt/share/DeepDXF_CGMN/data/CFG/OpenSSL_11ACFG_min10_max10/acfgSSL_11/CGMN.json'  # 修改为您希望保存 JSON 文件的路径

    # 调用 dxf_process_DeepDXF.py 的函数，处理 DXF 文件并生成 H5 文件
    print("开始处理 DXF 文件，生成 H5 文件...")
    process_dxf_files_for_deepdxf(input_dir, output_h5_dir)
    print("H5 文件生成完成。")

    # 调用 dxf_process_CGMN.py 的函数，处理 DXF 文件并生成 JSON 文件
    print("开始处理 DXF 文件，生成 JSON 文件...")
    process_dxf_files_for_cgmn(input_dir, output_json_file)
    print("JSON 文件生成完成。")

if __name__ == '__main__':
    main()
