import json
import os

# 文件路径
input_file = '/home/vllm/encode/data/DeepDXF/DeepDXF_truth.json'
output_file = '/home/vllm/encode/data/DeepDXF/DeepDXF_truth1.json'
test_directory = '/home/vllm/encode/data/DeepDXF/TEST_4096'

# 读取原始 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历数据中的每个文件列表，并检查文件是否存在
filtered_data = [
    [filename for filename in sublist if os.path.exists(os.path.join(test_directory, filename))]
    for sublist in data
]

# 将过滤后的数据写入到新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"新文件已保存为 {output_file}")
