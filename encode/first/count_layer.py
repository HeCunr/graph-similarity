# !/user/bin/env python3
# -*- coding: utf-8 -*-
import ezdxf
from collections import defaultdict

def count_entities_by_layer(file_path):
    # 读取DXF文件
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    # 创建一个字典以存储图层的实体类别和数量
    layer_entity_count = defaultdict(lambda: defaultdict(int))

    # 遍历模型空间中的所有实体
    for entity in msp:
        layer_name = entity.dxf.layer
        entity_type = entity.dxftype()
        layer_entity_count[layer_name][entity_type] += 1  # 增加该类别的计数

    # 输出结果
    for layer, entities in layer_entity_count.items():
        print(f"图层: {layer}")
        for entity_type, count in entities.items():
            print(f"  {entity_type}: {count}")

# 示例使用
file_path = r'C:\srtp\encode\datasets\4.dxf'
count_entities_by_layer(file_path)
