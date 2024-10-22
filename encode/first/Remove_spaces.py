import json
import os
import chardet

def remove_spaces_from_json(input_json):
    if isinstance(input_json, dict):
        if 'src' in input_json:
            input_json['src'] = input_json['src'].replace(' ', '')
        if 'fname' in input_json:
            input_json['fname'] = input_json['fname'].replace(' ', '')
    elif isinstance(input_json, list):
        for item in input_json:
            if 'src' in item:
                item['src'] = item['src'].replace(' ', '')
            if 'fname' in item:
                item['fname'] = item['fname'].replace(' ', '')
    return input_json

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(100000)  # 读取文件的一部分
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"检测到的文件编码：{encoding}，置信度：{confidence}")
    return encoding

if __name__ == '__main__':
    input_file = r'C:\Users\15653\dwg-cx\dataset\modified\split_by_own.json'    # 输入文件路径
    output_file =r'C:\Users\15653\dwg-cx\dataset\modified\split_by_own_remove_space.json'   # 输出文件路径

    # 检测文件编码
    encoding = detect_file_encoding(input_file)

    # 如果检测到的编码为 None，或者置信度较低，可以尝试手动指定编码
    if not encoding or encoding.lower() in ['ascii', 'unknown']:
        encoding = 'gbk'  # 根据实际情况选择合适的编码，如 'gbk'、'latin-1'、'utf-8'

    # 从文件加载JSON数据
    try:
        with open(input_file, 'r', encoding=encoding) as f:
            data = json.load(f)
    except UnicodeDecodeError as e:
        print(f"使用编码 {encoding} 读取文件时出错：{e}")
        # 尝试使用 'latin-1' 编码
        try:
            with open(input_file, 'r', encoding='latin-1') as f:
                data = json.load(f)
            print("使用 'latin-1' 编码成功读取文件。")
        except Exception as e:
            print(f"使用 'latin-1' 编码读取文件仍然出错：{e}")
            exit(1)
    except json.JSONDecodeError as e:
        print(f"解析JSON时出错: {e}")
        exit(1)

    # 移除 "src" 和 "fname" 中的空格
    modified_data = remove_spaces_from_json(data)

    # 将修改后的JSON数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=4, ensure_ascii=False)

    print(f"修改后的数据已写入 '{output_file}'。")

