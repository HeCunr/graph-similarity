# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import ezdxf
import numpy as np
from collections import defaultdict
from rtree import index
import shapely.geometry as geometry

# 实体类型列表
ENTITY_TYPES = ['ARC', 'TEXT', 'MTEXT', 'LWPOLYLINE', 'INSERT',
                'DIMENSION', 'LEADER', 'CIRCLE', 'HATCH', 'LINE']
ENTITY_LIMIT = 4096

def safe_get_coords(point):
    """安全地获取坐标，忽略Z值"""
    if isinstance(point, (tuple, list)):
        return point[:2]
    return point.x, point.y

def compute_entity_bounding_box(entity, doc):
    """计算单个实体的外框坐标"""
    entity_type = entity.dxftype()
    points = []

    try:
        if entity_type == 'LINE':
            points = [safe_get_coords(entity.dxf.start), safe_get_coords(entity.dxf.end)]
        elif entity_type == 'CIRCLE':
            center = safe_get_coords(entity.dxf.center)
            radius = entity.dxf.radius
            min_x, max_x = center[0] - radius, center[0] + radius
            min_y, max_y = center[1] - radius, center[1] + radius
            return min_x, min_y, max_x, max_y
        elif entity_type == 'ARC':
            center = safe_get_coords(entity.dxf.center)
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            points = [
                (center[0] + radius * np.cos(np.radians(angle)),
                 center[1] + radius * np.sin(np.radians(angle)))
                for angle in np.linspace(start_angle, end_angle, 100)
            ]
        elif entity_type in ['TEXT', 'MTEXT']:
            insert_point = safe_get_coords(entity.dxf.insert)
            height = getattr(entity.dxf, 'height', 0) or getattr(entity.dxf, 'char_height', 0)
            width = getattr(entity.dxf, 'width', 0)
            return insert_point[0], insert_point[1], insert_point[0] + width, insert_point[1] + height
        elif entity_type == 'LWPOLYLINE':
            points = [safe_get_coords(p) for p in entity.get_points()]
        elif entity_type == 'INSERT':
            insert_point = safe_get_coords(entity.dxf.insert)
            x_scale = getattr(entity.dxf, 'xscale', 1.0)
            y_scale = getattr(entity.dxf, 'yscale', 1.0)
            rotation = getattr(entity.dxf, 'rotation', 0.0)
            block = doc.blocks.get(entity.dxf.name)
            block_points = []
            for block_entity in block:
                bbox = compute_entity_bounding_box(block_entity, doc)
                if bbox:
                    block_points.extend([(bbox[0], bbox[1]), (bbox[2], bbox[3])])
            if block_points:
                min_x = min(p[0] for p in block_points)
                max_x = max(p[0] for p in block_points)
                min_y = min(p[1] for p in block_points)
                max_y = max(p[1] for p in block_points)
                center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
                transformed_points = []
                for x, y in block_points:
                    x, y = x - center_x, y - center_y
                    x, y = x * x_scale, y * y_scale
                    x_new = x * np.cos(np.radians(rotation)) - y * np.sin(np.radians(rotation))
                    y_new = x * np.sin(np.radians(rotation)) + y * np.cos(np.radians(rotation))
                    x_new, y_new = x_new + insert_point[0], y_new + insert_point[1]
                    transformed_points.append((x_new, y_new))
                return min(p[0] for p in transformed_points), min(p[1] for p in transformed_points), \
                       max(p[0] for p in transformed_points), max(p[1] for p in transformed_points)
        elif entity_type in ['DIMENSION', 'LEADER', 'HATCH']:
            if entity_type == 'DIMENSION':
                points = [safe_get_coords(entity.dxf.defpoint), safe_get_coords(entity.dxf.text_midpoint)]
                if hasattr(entity.dxf, 'defpoint2'):
                    points.append(safe_get_coords(entity.dxf.defpoint2))
                if hasattr(entity.dxf, 'defpoint3'):
                    points.append(safe_get_coords(entity.dxf.defpoint3))
            elif entity_type == 'LEADER':
                points = [safe_get_coords(v) for v in entity.vertices]
            elif entity_type == 'HATCH':
                for path in entity.paths:
                    # 修改这部分代码来正确处理HATCH实体
                    if hasattr(path, 'edges'):  # EdgePath
                        for edge in path.edges:
                            if hasattr(edge, 'start') and hasattr(edge, 'end'):
                                points.extend([safe_get_coords(edge.start), safe_get_coords(edge.end)])
                            elif hasattr(edge, 'center') and hasattr(edge, 'radius'):
                                center = safe_get_coords(edge.center)
                                radius = edge.radius
                                # 对于圆弧，采样更多点以提高精度
                                angles = np.linspace(0, 360, 36)  # 每10度取一个点
                                points.extend([
                                    (center[0] + radius * np.cos(np.radians(angle)),
                                     center[1] + radius * np.sin(np.radians(angle)))
                                    for angle in angles
                                ])
                    elif hasattr(path, 'vertices'):  # PolylinePath
                        points.extend([safe_get_coords(vertex) for vertex in path.vertices])

        if points:
            xs, ys = zip(*points)
            return min(xs), min(ys), max(xs), max(ys)
        else:
            return None

    except Exception as e:
        print(f"Error processing entity {entity.dxf.handle} of type {entity_type}: {e}")
        return None

def check_overlap(bbox1, bbox2):
    """检查两个边界框是否重叠或接触，但不包括完全包含的情况
    Args:
        bbox1: 第一个边界框的坐标 (min_x, min_y, max_x, max_y)
        bbox2: 第二个边界框的坐标 (min_x, min_y, max_x, max_y)
    Returns:
        bool: 如果边界框重叠或接触（不包括完全包含）则返回True，否则返回False
    """
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    # 检查是否完全没有接触（不相交）
    if max_x1 < min_x2 or min_x1 > max_x2:  # x轴方向无交集
        return False
    if max_y1 < min_y2 or min_y1 > max_y2:  # y轴方向无交集
        return False

    # 检查是否存在包含关系
    box1_contains_box2 = (min_x1 <= min_x2 and max_x1 >= max_x2 and
                          min_y1 <= min_y2 and max_y1 >= max_y2)
    box2_contains_box1 = (min_x2 <= min_x1 and max_x2 >= max_x1 and
                          min_y2 <= min_y1 and max_y2 >= max_y1)

    # 如果存在包含关系，返回False
    if box1_contains_box2 or box2_contains_box1:
        return False

    # 其他情况（有重叠、边缘接触或顶点接触）返回True
    return True

def get_normalized_entity_handles(msp):
    entity_handles = []
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type in ENTITY_TYPES:
            handle = int(entity.dxf.handle, 16)  # 将十六进制转换为十进制
            entity_handles.append((entity_type, handle, entity))

    # 按句柄值排序
    entity_handles.sort(key=lambda x: x[1])

    return entity_handles

def create_adjacency_list(entity_handles):
    """创建邻接表"""
    adjacency_list = [[] for _ in range(len(ENTITY_TYPES))]
    type_to_index = {t: i for i, t in enumerate(ENTITY_TYPES)}

    for i in range(len(entity_handles)):
        current_type = entity_handles[i][0]
        current_index = type_to_index[current_type]

        # 前一个实体
        if i > 0:
            prev_type = entity_handles[i - 1][0]
            prev_index = type_to_index[prev_type]
            if prev_index != current_index and prev_index not in adjacency_list[current_index]:
                adjacency_list[current_index].append(prev_index)

        # 后一个实体
        if i < len(entity_handles) - 1:
            next_type = entity_handles[i + 1][0]
            next_index = type_to_index[next_type]
            if next_index != current_index and next_index not in adjacency_list[current_index]:
                adjacency_list[current_index].append(next_index)

    return adjacency_list

def process_dxf_file(file_path):
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()

        # 获取实体句柄并排序
        entity_handles = get_normalized_entity_handles(msp)

        # 计算实体总数
        entity_count = len(entity_handles)

        # 如果实体数量超过限制，返回None
        if entity_count > ENTITY_LIMIT:
            print(f"Skipping {file_path}: Entity count ({entity_count}) exceeds limit ({ENTITY_LIMIT})")
            return None

        # 初始化数据结构
        entities = []
        total_counts = {etype: 0 for etype in ENTITY_TYPES}
        overlapping_counts = {etype: {other_type: 0 for other_type in ENTITY_TYPES} for etype in ENTITY_TYPES}

        # 创建R树索引
        idx = index.Index()

        # 第一遍遍历：收集实体信息并插入R树
        for entity_id, (entity_type, handle, entity) in enumerate(entity_handles):
            bbox = compute_entity_bounding_box(entity, doc)
            if bbox:
                entities.append({
                    'type': entity_type,
                    'bbox': bbox,
                    'id': entity_id
                })
                total_counts[entity_type] += 1
                idx.insert(entity_id, bbox)

        # 第二遍遍历：检查重叠
        for entity in entities:
            etype = entity['type']
            bbox = entity['bbox']
            entity_id = entity['id']

            possible_matches = list(idx.intersection(bbox))
            for match_id in possible_matches:
                if match_id == entity_id:
                    continue
                other_entity = entities[match_id]
                other_type = other_entity['type']

                if other_entity['id'] < entity_id:
                    continue

                if check_overlap(bbox, other_entity['bbox']):
                    overlapping_counts[etype][other_type] += 1
                    overlapping_counts[other_type][etype] += 1

        # 生成特征向量
        feature_vectors = {}
        for etype in ENTITY_TYPES:
            vector = [total_counts[etype] - sum(overlapping_counts[etype].values())]  # SELF
            for other_type in ENTITY_TYPES:
                if other_type == etype:
                    vector.insert(1, 0)  # 自身类型的重叠数量设为0，放在第一列之后
                else:
                    vector.append(overlapping_counts[etype][other_type])
            feature_vectors[etype] = vector

        # 归一化特征向量
        normalized_vectors = normalize_feature_vectors(feature_vectors)

        # 创建邻接表
        adjacency_list = create_adjacency_list(entity_handles)

        # 获取文件名并去除中间的空格
        base_filename = os.path.basename(file_path)
        fname = base_filename.replace(' ', '')

        return {
            "src": base_filename,
            "n_num": len(ENTITY_TYPES),
            "succs": adjacency_list,
            "features": normalized_vectors,
            "fname": os.path.splitext(fname)[0]
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def normalize_feature_vectors(feature_vectors):
    """对特征向量进行归一化"""
    normalized_vectors = []
    first_column = [feature_vectors[etype][0] for etype in ENTITY_TYPES]
    first_col_sum = sum(first_column)

    for etype in ENTITY_TYPES:
        vector = feature_vectors[etype]
        normalized_vector = []

        # 归一化第一列（自身类型非重叠数量）
        normalized_self = vector[0] / first_col_sum if first_col_sum != 0 else 0
        normalized_vector.append(normalized_self)

        # 归一化自身类型重叠数量（为0）
        normalized_vector.append(0)

        # 归一化其余列（与其他类型的重叠数量），按行归一化
        other_values = vector[2:]
        row_sum = sum(other_values)
        if row_sum != 0:
            normalized_others = [value / row_sum for value in other_values]
        else:
            normalized_others = [0] * len(other_values)
        normalized_vector.extend(normalized_others)

        normalized_vectors.append(normalized_vector)

    return normalized_vectors

def process_directory(input_directory, output_file_path):
    """处理指定目录下的所有DXF文件并输出结果"""
    with open(output_file_path, 'w') as output_file:
        for filename in os.listdir(input_directory):
            if filename.lower().endswith('.dxf'):
                file_path = os.path.join(input_directory, filename)
                result = process_dxf_file(file_path)
                if result is not None:
                    json_line = json.dumps(result)
                    output_file.write(json_line + '\n')
                    print(f"Processed: {filename}")
                else:
                    print(f"Skipped: {filename}")
    print(f"Results written to {output_file_path}")


def process_dxf_files_for_cgmn(input_directory, output_file_path):
    """处理指定目录下的所有DXF文件并输出结果"""
    with open(output_file_path, 'w') as output_file:
        for filename in os.listdir(input_directory):
            if filename.lower().endswith('.dxf'):
                file_path = os.path.join(input_directory, filename)
                result = process_dxf_file(file_path)
                if result is not None:
                    json_line = json.dumps(result)
                    output_file.write(json_line + '\n')
                    print(f"Processed: {filename}")
                else:
                    print(f"Skipped: {filename}")
    print(f"Results written to {output_file_path}")

if __name__ == '__main__':
    # 示例用法
    input_directory =  r'/mnt/share/DeepDXF_CGMN/encode/data/241101'  # 修改为您的 DXF 文件目录
    output_file_path = r'/mnt/share/DeepDXF_CGMN/encode/data/GF/4096.json'  # 修改为您希望保存 JSON 文件的路径
    process_dxf_files_for_cgmn(input_directory, output_file_path)

