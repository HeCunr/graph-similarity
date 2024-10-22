# !/user/bin/env python3
# -*- coding: utf-8 -*-
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import ezdxf
import math
import os
import json

def get_entity_points(entity):
    """获取实体的所有坐标点，用于计算边界框"""
    entity_type = entity.dxftype()
    points = []

    if entity_type == 'LINE':
        points.append(entity.dxf.start)
        points.append(entity.dxf.end)
    elif entity_type == 'LWPOLYLINE':
        points.extend([point[:2] for point in entity.get_points()])
    elif entity_type == 'POLYLINE':
        points.extend([vertex.dxf.location for vertex in entity.vertices])
    elif entity_type == 'CIRCLE':
        center = entity.dxf.center
        radius = entity.dxf.radius
        # 使用圆心和半径计算边界点
        points.append((center[0] - radius, center[1]))
        points.append((center[0] + radius, center[1]))
        points.append((center[0], center[1] - radius))
        points.append((center[0], center[1] + radius))
    elif entity_type == 'ARC':
        center = entity.dxf.center
        radius = entity.dxf.radius
        # 近似处理弧的边界
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        angles = [start_angle, end_angle]
        for angle in angles:
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
    elif entity_type in ['TEXT', 'MTEXT']:
        insert_point = entity.dxf.insert
        points.append(insert_point)
    elif entity_type == 'HATCH':
        for path in entity.paths:
            if path.PATH_TYPE == 'EdgePath':
                for edge in path.edges:
                    if edge.EDGE_TYPE == 'LineEdge':
                        points.append(edge.start)
                        points.append(edge.end)
                    elif edge.EDGE_TYPE == 'ArcEdge':
                        # 可根据需要添加更多处理
                        pass
    elif entity_type == 'DIMENSION':
        defpoint = entity.dxf.defpoint
        text_midpoint = entity.dxf.text_midpoint
        points.append(defpoint)
        points.append(text_midpoint)
    elif entity_type == 'LEADER':
        points.extend(entity.vertices)
    elif entity_type == 'INSERT':
        insert_point = entity.dxf.insert
        points.append(insert_point)

    return points

def get_bounding_box(doc):
    """计算模型的边界框"""
    msp = doc.modelspace()
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for entity in msp:
        try:
            points = get_entity_points(entity)
            for point in points:
                x, y = point[0], point[1]
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        except Exception as e:
            print(f"无法处理实体 {entity.dxftype()}：{e}")

    return min_x, min_y, max_x, max_y

def normalize_coordinate(value, min_val, max_val):
    """将坐标归一化到 [-1, 1] 范围"""
    return 2 * (value - min_val) / (max_val - min_val) - 1

def normalize_length(value, max_dim):
    """根据模型最大尺寸归一化长度参数"""
    return value * (2 / max_dim)

def process_dxf_file(input_dxf_path, output_dir):
    doc = ezdxf.readfile(input_dxf_path)
    msp = doc.modelspace()

    min_x, min_y, max_x, max_y = get_bounding_box(doc)
    print(f"边界框：min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")

    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)

    entities = []
    for entity in msp:
        entity_type = entity.dxftype()
        features = {}

        if entity_type == 'LINE':
            # LINE 特征
            start_point = entity.dxf.start
            end_point = entity.dxf.end
            features['start_point'] = [
                normalize_coordinate(start_point[0], min_x, max_x),
                normalize_coordinate(start_point[1], min_y, max_y)
            ]
            features['end_point'] = [
                normalize_coordinate(end_point[0], min_x, max_x),
                normalize_coordinate(end_point[1], min_y, max_y)
            ]

        elif entity_type == 'CIRCLE':
            # CIRCLE 特征
            center = entity.dxf.center
            radius = entity.dxf.radius
            features['center'] = [
                normalize_coordinate(center[0], min_x, max_x),
                normalize_coordinate(center[1], min_y, max_y)
            ]
            features['radius'] = normalize_length(radius, max_dim)

        elif entity_type == 'ARC':
            # ARC 特征
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            features['center'] = [
                normalize_coordinate(center[0], min_x, max_x),
                normalize_coordinate(center[1], min_y, max_y)
            ]
            features['radius'] = normalize_length(radius, max_dim)
            features['start_angle'] = start_angle  # 角度不进行缩放
            features['end_angle'] = end_angle

        elif entity_type == 'LWPOLYLINE':
            # LWPOLYLINE 特征
            closed = entity.closed
            points = [point[:2] for point in entity.get_points()]
            normalized_points = [
                [
                    normalize_coordinate(p[0], min_x, max_x),
                    normalize_coordinate(p[1], min_y, max_y)
                ] for p in points
            ]
            features['closed'] = closed
            features['points'] = normalized_points
            features['count'] = len(points)

        elif entity_type == 'TEXT':
            # TEXT 特征
            insert_point = entity.dxf.insert
            height = entity.dxf.height
            rotation = entity.dxf.rotation
            text = entity.dxf.text
            features['insert_point'] = [
                normalize_coordinate(insert_point[0], min_x, max_x),
                normalize_coordinate(insert_point[1], min_y, max_y)
            ]
            features['height'] = normalize_length(height, max_dim)
            features['rotation'] = rotation  # 角度不进行缩放
            features['text'] = text  # 名称和文本不处理

        elif entity_type == 'MTEXT':
            # MTEXT 特征
            insert_point = entity.dxf.insert
            char_height = entity.dxf.char_height
            width = entity.dxf.width
            rotation = entity.dxf.rotation
            text = entity.text
            features['insert_point'] = [
                normalize_coordinate(insert_point[0], min_x, max_x),
                normalize_coordinate(insert_point[1], min_y, max_y)
            ]
            features['char_height'] = normalize_length(char_height, max_dim)
            features['width'] = normalize_length(width, max_dim)
            features['rotation'] = rotation
            features['text'] = text  # 名称和文本不处理

        elif entity_type == 'HATCH':
            # HATCH 特征
            solid_fill = entity.dxf.solid_fill
            associative = entity.dxf.associative
            boundary_paths = len(entity.paths)
            pattern_name = entity.dxf.pattern_name
            features['solid_fill'] = solid_fill
            features['associative'] = associative
            features['boundary_paths'] = boundary_paths
            features['pattern_name'] = pattern_name  # 名称和文本不处理

        elif entity_type == 'DIMENSION':
            # DIMENSION 特征
            defpoint = entity.dxf.defpoint
            text_midpoint = entity.dxf.text_midpoint
            dim_type = entity.dxf.dimtype
            features['defpoint'] = [
                normalize_coordinate(defpoint[0], min_x, max_x),
                normalize_coordinate(defpoint[1], min_y, max_y)
            ]
            features['text_midpoint'] = [
                normalize_coordinate(text_midpoint[0], min_x, max_x),
                normalize_coordinate(text_midpoint[1], min_y, max_y)
            ]
            features['dim_type'] = dim_type

        elif entity_type == 'LEADER':
            # LEADER 特征
            vertices = entity.vertices
            normalized_vertices = [
                [
                    normalize_coordinate(v[0], min_x, max_x),
                    normalize_coordinate(v[1], min_y, max_y)
                ] for v in vertices
            ]
            annotation_type = entity.dxf.annotation_type
            features['vertices'] = normalized_vertices
            features['annotation_type'] = annotation_type

        elif entity_type == 'INSERT':
            # INSERT 特征
            name = entity.dxf.name
            insert_point = entity.dxf.insert
            scale = (
                entity.dxf.xscale,
                entity.dxf.yscale,
                entity.dxf.zscale
            )
            rotation = entity.dxf.rotation
            features['name'] = name  # 名称和文本不处理
            features['insert_point'] = [
                normalize_coordinate(insert_point[0], min_x, max_x),
                normalize_coordinate(insert_point[1], min_y, max_y)
            ]
            features['scale'] = scale  # 缩放比例不进行处理
            features['rotation'] = rotation

        else:
            # 其他实体类型可根据需要添加
            continue

        entity_dict = {
            'type': entity_type,
            'features': features
        }
        entities.append(entity_dict)

    # 生成输出 JSON 文件路径
    input_filename = os.path.basename(input_dxf_path)
    output_filename = os.path.splitext(input_filename)[0] + '.json'
    output_json_path = os.path.join(output_dir, output_filename)

    # 保存处理后的实体到 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(entities, f, indent=2)

    print(f"处理完成，结果已保存到 {output_json_path}")

def main():
    input_dir = r'C:\srtp\encode\data\dxf'  # 请替换为包含 DXF 文件的输入目录路径
    output_dir = r'C:\srtp\encode\data\raw'  # 请替换为保存 JSON 文件的输出目录路径

    # 获取输入目录下的所有文件列表
    files = os.listdir(input_dir)

    # 遍历文件列表，处理每个 DXF 文件
    for file in files:
        if file.endswith('.dxf'):
            input_dxf_path = os.path.join(input_dir, file)
            process_dxf_file(input_dxf_path, output_dir)

if __name__ == "__main__":
    main()