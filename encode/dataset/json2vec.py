

import os
import numpy as np
import h5py
from dataset.DeepDXF_json_dataset import load_dxf

def process_dxf(dxf_file):
    dxf_vec = load_dxf(dxf_file)
    return dxf_vec

def main():
    input_dir = r'C:\srtp\encode\data\DeepDXF\dxf_json'  # 指定输入目录
    output_dir = r'C:\srtp\encode\data\DeepDXF\dxf_vec'  # 指定输出目录

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中所有的 JSON 文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)

        # 为每个输入文件创建对应的输出文件名
        output_file = os.path.splitext(json_file)[0] + '.h5'
        output_path = os.path.join(output_dir, output_file)

        # 处理 DXF 文件
        dxf_vec = process_dxf(input_path)

        if dxf_vec is not None:
            # 创建 H5 文件并保存向量数据
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('dxf_vec', data=np.array([dxf_vec]), dtype=np.int16)

            print(f"Processed {json_file} and saved to {output_file}")
        else:
            print(f"Failed to process {json_file}")

if __name__ == '__main__':
    main()