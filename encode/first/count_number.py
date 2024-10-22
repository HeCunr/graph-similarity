import ezdxf
from collections import defaultdict

def count_dxf_entities(file_path):
    # 创建一个字典来存储实体类别的计数
    entity_count = defaultdict(int)

    # 读取DXF文件
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    # 遍历模型空间中的所有实体
    for entity in msp:
        entity_type = entity.dxftype()
        entity_count[entity_type] += 1  # 计数该实体类型

    return entity_count

# 示例使用
file_path = r'C:\srtp\encode\datasets\6.dxf'
entity_statistics = count_dxf_entities(file_path)

# 输出实体类别及其数量
for entity_type, count in entity_statistics.items():
    print(f"{entity_type}: {count}")




