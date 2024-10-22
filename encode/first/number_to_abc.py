import json
import chardet

def digit_to_word(digit_str):
    mapping = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }
    return mapping.get(digit_str, digit_str)

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        data = f.read(10000)
        result = chardet.detect(data)
        return result['encoding']

def convert_fname_in_json_file(input_txt_file, output_txt_file):
    encoding = detect_encoding(input_txt_file)
    print(f"检测到的文件编码: {encoding}")

    modified_data = []

    with open(input_txt_file, 'r', encoding=encoding) as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误（第 {line_number} 行）：{e}")
                continue

            if 'fname' in data:
                original_fname = data['fname']
                data['fname'] = digit_to_word(data['fname'])
                print(f"将 'fname' 从 '{original_fname}' 转换为 '{data['fname']}'")

            modified_data.append(data)

    # 将所有修改后的数据写入输出文件
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for item in modified_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"转换完成，结果已保存到 {output_txt_file}")

if __name__ == '__main__':
    input_txt_file = r'C:\Users\15653\dwg-cx\dataset\modified\test_by_5class.txt'    # 替换为您的输入文件路径
    output_txt_file =r'C:\Users\15653\dwg-cx\dataset\modified\test_by_5class_abc.txt' # 替换为您的输出文件路径
    convert_fname_in_json_file(input_txt_file, output_txt_file)
