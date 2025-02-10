import os
import json

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存 JSON 数据到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def filter_h5_files(json_data, directory):
    """过滤掉不在指定目录中的 h5 文件"""
    filtered_data = []

    # 获取目录中的所有 h5 文件名
    h5_files_in_dir = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                h5_files_in_dir.add(file)

    # 遍历 JSON 数据中的每个列表
    for h5_list in json_data:
        filtered_list = [h5_file for h5_file in h5_list if h5_file in h5_files_in_dir]
        if filtered_list:  # 只保留非空列表
            filtered_data.append(filtered_list)

    return filtered_data

if __name__ == "__main__":
    # 输入文件路径
    input_json_path = '/home/vllm/encode/data/Geom/Geom_truth.json'
    output_json_path = '/home/vllm/encode/data/Geom/Geom_truth_2048.json'
    target_directory = '/home/vllm/encode/data/Geom/TEST_2048'

    # 检查输入文件和目标目录是否存在
    if not os.path.isfile(input_json_path):
        print(f"错误：输入文件 '{input_json_path}' 不存在。")
        exit(1)

    if not os.path.isdir(target_directory):
        print(f"错误：目标目录 '{target_directory}' 不存在。")
        exit(1)

    # 加载原始 JSON 数据
    try:
        json_data = load_json(input_json_path)
    except Exception as e:
        print(f"错误：无法加载 JSON 文件。{e}")
        exit(1)

    # 过滤 h5 文件
    filtered_data = filter_h5_files(json_data, target_directory)

    # 保存过滤后的 JSON 数据
    save_json(filtered_data, output_json_path)
    print(f"处理完成，结果已保存到 '{output_json_path}'。")