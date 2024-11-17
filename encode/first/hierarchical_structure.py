import ezdxf
from collections import defaultdict

def parse_dxf_structure(file_path):
    # 读取DXF文件
    doc = ezdxf.readfile(file_path)

    # 提取图层信息
    layers = doc.layers
    layer_info = {layer.dxf.name: layer.dxf.color for layer in layers}

    # 提取块信息及其中的实体类别
    blocks = doc.blocks

    block_info = {}
    for block in blocks:
        entity_count = defaultdict(int)  # 用于统计每个块内的实体类别
        for entity in block:
            entity_count[entity.dxftype()] += 1
        block_info[block.name] = dict(entity_count)  # 存储块名及其实体类别数量


    # 提取模型空间中的实体
    msp = doc.modelspace()
    entity_count = {entity.dxftype(): 0 for entity in msp}
    for entity in msp:
        entity_count[entity.dxftype()] += 1



    return {
        'layers': layer_info,
        'blocks': block_info,
        'entities': entity_count
    }

# 示例使用
file_path =  r'C:\srtp\241101\DFN1006-2, 78W, 0.508 X 0.508 PAD SIZE LF(4)_1.dxf' # 请替换为您的DXF文件路径
#file_path = r'C:\srtp\dwg\2018_Binary_DXF\8-040116-SOP1(ETCH) REV A.dxf'  # 请替换为您的DXF文件路径
dxf_structure = parse_dxf_structure(file_path)

# 输出解析结果
print("图层信息:")
print(dxf_structure['layers'])
print("\n块信息（实体类别及数量）:")
for block_name, entities in dxf_structure['blocks'].items():
    print(f"{block_name}: {entities}")
print("\n实体数量:")
print(dxf_structure['entities'])

