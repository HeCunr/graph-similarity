# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import re
from convert_edge import *

def extract_file_names(directory):
    # 提取目录下所有文件的名称
    file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return file_names

def remove_numbers(file_name):
    # 去除文件名称中的数字
    return re.sub(r'\d+', '', file_name)

def save_to_json(file_data, output_file):
    # 保存数据为JSON格式
    with open(output_file, 'w') as json_file:
        json.dump(file_data, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    directory = r'C:\srtp\encode\datasets\test'
    output_file = r'C:\srtp\encode\datasets\test\file_names.json'

    # 提取文件名称
    file_names = extract_file_names(directory)

    # 构建JSON数据
    json_data = []
    for file_name in file_names:
        fname = remove_numbers(file_name)  # 去除数字
        json_data.append({
            "src": file_name,
            "n_num":2,
            "succs":3,
            "features":3,
            "fname": fname
        })

    # 保存为JSON文件
    save_to_json(json_data, output_file)
    print(f"文件名称已保存至 {output_file}")
