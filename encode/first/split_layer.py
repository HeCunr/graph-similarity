import ezdxf

def copy_entity(entity, new_msp):
    """复制实体的属性到新的模型空间"""
    if entity.dxftype() == 'LINE':
        new_msp.add_line(entity.dxf.start, entity.dxf.end, dxfattribs={'layer': entity.dxf.layer})
    elif entity.dxftype() == 'CIRCLE':
        new_msp.add_circle(entity.dxf.center, entity.dxf.radius, dxfattribs={'layer': entity.dxf.layer})
    elif entity.dxftype() == 'LWPOLYLINE':
        vertices = entity.vertices()
        new_msp.add_lwpolyline(vertices, dxfattribs={'layer': entity.dxf.layer})
    # elif entity.dxftype() == 'TEXT':
    #     new_msp.add_text(entity.dxf.text, insert=entity.dxf.insert, dxfattribs={'layer': entity.dxf.layer})
    # elif entity.dxftype() == 'MTEXT':
    #     new_msp.add_mtext(entity.text, insert=entity.dxf.insert, dxfattribs={'layer': entity.dxf.layer})
    # elif entity.dxftype() == 'HATCH':
    #     new_hatch = new_msp.add_hatch(0, dxfattribs={'layer': entity.dxf.layer})
    #     new_hatch.set_pattern(entity.pattern_name, entity.pattern_scale)
    elif entity.dxftype() == 'INSERT':
        new_msp.add_blockref(entity.dxf.name, insert=entity.dxf.insert, dxfattribs={'layer': entity.dxf.layer})
    # elif entity.dxftype() == 'LEADER':
    #     points = entity.get_points()  # 获取LEADER的所有点
    #     new_msp.add_leader(start_point=entity.dxf.start_point, points=points, dxfattribs={'layer': entity.dxf.layer})
    # elif entity.dxftype() == 'DIMENSION':
    #     # 注意：DIMENSION的具体实现需要根据类型进行详细处理
    #     new_msp.add_dimension(entity.dxf.type, entity.dxf.insert, dxfattribs={'layer': entity.dxf.layer})

def split_dxf_by_layers(file_path):
    # 读取原始DXF文件
    doc = ezdxf.readfile(file_path)

    # 获取所有图层
    layers = doc.layers

    # 创建一个字典来存储每个图层的实体
    layer_entities = {layer.dxf.name: [] for layer in layers}

    # 遍历模型空间中的所有实体
    msp = doc.modelspace()
    for entity in msp:
        layer_name = entity.dxf.layer
        layer_entities[layer_name].append(entity)  # 将实体添加到对应图层

    # 为每个图层创建一个新的DXF文件
    for layer_name, entities in layer_entities.items():
        if entities:  # 如果该图层有实体
            # 创建新的DXF文档
            new_doc = ezdxf.new()
            new_msp = new_doc.modelspace()

            # 将该图层的实体添加到新文档中
            for entity in entities:
                copy_entity(entity, new_msp)

            # 保存新的DXF文件
            new_file_name = f"{layer_name}.dxf"
            new_doc.saveas(new_file_name)
            print(f"已保存图层 '{layer_name}' 到文件: {new_file_name}")

# 示例使用
file_path = r'C:\srtp\encode\datasets\3.dxf'
split_dxf_by_layers(file_path)
