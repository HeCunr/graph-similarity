import json
import os

def convert_and_save(input_path, output_path):
    try:
        # 读取源文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换格式：移除每个文件名中的.json后缀
        converted_data = []
        for file_list in data:
            converted_sublist = []
            for filename in file_list:
                # 移除.json后缀
                name_without_ext = filename.replace('.json', '')
                converted_sublist.append(name_without_ext)
            converted_data.append(converted_sublist)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 将转换后的数据格式化为字符串
        output_content = '['
        for i, sublist in enumerate(converted_data):
            output_content += '['
            for j, filename in enumerate(sublist):
                output_content += f'"{filename}"'
                if j < len(sublist) - 1:
                    output_content += ',\n'
            output_content += ']'
            if i < len(converted_data) - 1:
                output_content += ',\n\n'
        output_content += ']'

        # 保存为txt文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)

        print(f"转换成功! 文件已保存到: {output_path}")

    except FileNotFoundError:
        print("错误: 找不到输入文件")
    except json.JSONDecodeError:
        print("错误: 输入文件格式不正确")
    except Exception as e:
        print(f"发生错误: {str(e)}")

# 使用示例
input_file = r"C:\srtp\FIRST PAPER\encode\data\Geom\Geom_truth.json"  # 替换为实际的输入文件路径
output_file = r"C:\srtp\FIRST PAPER\MulConDXF\data\truth.txt"  # 替换为实际的输出文件路径

convert_and_save(input_file, output_file)