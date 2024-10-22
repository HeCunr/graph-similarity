#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ezdxf
import math
from ezdxf.entities import BoundaryPathType, EdgeType
from rtree import index  # 导入 R 树索引
import shapely.geometry as geometry  # 用于创建矩形

# 定义实体类型列表，移除了 'POLYLINE'
ENTITY_TYPES = ['ARC', 'TEXT', 'MTEXT',
                'LWPOLYLINE', 'INSERT',
                'DIMENSION', 'LEADER', 'CIRCLE', 'HATCH', 'LINE']

def compute_entity_bounding_box(entity, doc):
    """
    计算单个实体的外框坐标，返回 (min_x, min_y, max_x, max_y)
    """
    entity_type = entity.dxftype()
    points = []

    try:
        if entity_type == 'LINE':
            # 直线的起点和终点
            points = [entity.dxf.start, entity.dxf.end]

        elif entity_type == 'CIRCLE':
            # 圆的边界框为其外接正方形
            center = entity.dxf.center
            radius = entity.dxf.radius
            min_x = center[0] - radius
            max_x = center[0] + radius
            min_y = center[1] - radius
            max_y = center[1] + radius
            return min_x, min_y, max_x, max_y

        elif entity_type == 'ARC':
            # 计算圆弧的边界框，通过采样点近似计算
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = math.radians(entity.dxf.start_angle)
            end_angle = math.radians(entity.dxf.end_angle)

            # 处理跨越0度的情况
            if end_angle < start_angle:
                end_angle += 2 * math.pi

            num_points = 100
            angles = [start_angle + t * (end_angle - start_angle) / (num_points - 1) for t in range(num_points)]
            points = [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]

        elif entity_type in ['TEXT', 'MTEXT']:
            # 使用文本的插入点，假设文本是水平的，使用高度和宽度
            insert_point = entity.dxf.insert
            height = entity.dxf.height if hasattr(entity.dxf, 'height') else 0
            width = entity.dxf.width if hasattr(entity.dxf, 'width') else 0
            x, y = insert_point[0], insert_point[1]
            min_x = x
            max_x = x + width
            min_y = y
            max_y = y + height
            return min_x, min_y, max_x, max_y

        elif entity_type == 'LWPOLYLINE':
            # 轻量级多段线的所有顶点
            points = entity.get_points('xy')

        elif entity_type == 'SPLINE':
            # 样条曲线的拟合点或控制点
            if entity.has_fit_points:
                points = entity.fit_points
            else:
                points = entity.control_points

        elif entity_type == 'INSERT':
            # 计算块引用的外框
            insert_point = entity.dxf.insert
            x_scale = entity.dxf.xscale if hasattr(entity.dxf, 'xscale') else 1.0
            y_scale = entity.dxf.yscale if hasattr(entity.dxf, 'yscale') else 1.0
            rotation = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0.0

            block_name = entity.dxf.name
            block = doc.blocks.get(block_name)
            block_points = []

            for block_entity in block:
                bbox = compute_entity_bounding_box(block_entity, doc)
                if bbox:
                    min_x, min_y, max_x, max_y = bbox
                    block_points.extend([(min_x, min_y), (max_x, max_y)])

            if block_points:
                xs = [p[0] for p in block_points]
                ys = [p[1] for p in block_points]
                min_x_block = min(xs)
                max_x_block = max(xs)
                min_y_block = min(ys)
                max_y_block = max(ys)

                # 应用缩放和旋转
                # 首先，将块的边界中心移动到原点
                center_x = (min_x_block + max_x_block) / 2
                center_y = (min_y_block + max_y_block) / 2
                transformed_points = []

                for x, y in block_points:
                    # 平移到原点
                    x -= center_x
                    y -= center_y
                    # 缩放
                    x *= x_scale
                    y *= y_scale
                    # 旋转
                    angle = math.radians(rotation)
                    x_new = x * math.cos(angle) - y * math.sin(angle)
                    y_new = x * math.sin(angle) + y * math.cos(angle)
                    # 平移回插入点
                    x_new += insert_point[0]
                    y_new += insert_point[1]
                    transformed_points.append((x_new, y_new))

                xs = [p[0] for p in transformed_points]
                ys = [p[1] for p in transformed_points]
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                return min_x, min_y, max_x, max_y
            else:
                return None

        elif entity_type == 'DIMENSION':
            # 计算标注的外框
            dim_type = entity.dimtype
            dim_points = []

            # 获取定义点和中间点
            def_point = entity.dxf.defpoint  # 定义点
            text_midpoint = entity.dxf.text_midpoint  # 文字中点
            dim_points.append((def_point[0], def_point[1]))
            if text_midpoint:
                dim_points.append((text_midpoint[0], text_midpoint[1]))

            # 获取标注线的起点和终点
            if hasattr(entity.dxf, 'dim_line_point'):
                dim_line_point = entity.dxf.dim_line_point
                dim_points.append((dim_line_point[0], dim_line_point[1]))

            # 对于线性标注，获取扩展线的起点和终点
            if dim_type == 0:  # 线性标注
                if hasattr(entity.dxf, 'defpoint2'):
                    def_point2 = entity.dxf.defpoint2
                    dim_points.append((def_point2[0], def_point2[1]))
                if hasattr(entity.dxf, 'defpoint3'):
                    def_point3 = entity.dxf.defpoint3
                    dim_points.append((def_point3[0], def_point3[1]))

            if dim_points:
                xs = [p[0] for p in dim_points]
                ys = [p[1] for p in dim_points]
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                return min_x, min_y, max_x, max_y
            else:
                return None

        elif entity_type == 'LEADER':
            # 引线的所有顶点
            points = entity.vertices

        elif entity_type == 'HATCH':
            # 处理HATCH实体，提取其边界点
            for boundary_path in entity.paths:
                if boundary_path.type == BoundaryPathType.EDGE:
                    for edge in boundary_path.edges:
                        if edge.type == EdgeType.LINE:
                            # 直线边界，添加起点和终点
                            points.append(edge.start)
                            points.append(edge.end)
                        elif edge.type == EdgeType.ARC:
                            # 圆弧边界，采样弧上的点
                            center = edge.center
                            radius = edge.radius
                            start_angle = math.radians(edge.start_angle)
                            end_angle = math.radians(edge.end_angle)
                            if end_angle < start_angle:
                                end_angle += 2 * math.pi
                            num_points = 20
                            angles = [start_angle + t * (end_angle - start_angle) / (num_points - 1) for t in range(num_points)]
                            arc_points = [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]
                            points.extend(arc_points)
                        elif edge.type == EdgeType.ELLIPSE:
                            # 椭圆边界，采样椭圆上的点
                            center = edge.center
                            major_axis = edge.major_axis
                            ratio = edge.radius_ratio
                            start_param = edge.start_param
                            end_param = edge.end_param
                            if end_param < start_param:
                                end_param += 2 * math.pi
                            num_points = 100
                            params = [start_param + t * (end_param - start_param) / (num_points - 1) for t in range(num_points)]
                            ellipse_points = []
                            for param in params:
                                cos_param = math.cos(param)
                                sin_param = math.sin(param)
                                x = center[0] + major_axis[0] * cos_param - major_axis[1] * sin_param * ratio
                                y = center[1] + major_axis[1] * cos_param + major_axis[0] * sin_param * ratio
                                ellipse_points.append((x, y))
                            points.extend(ellipse_points)
                        elif edge.type == EdgeType.SPLINE:
                            # 样条曲线边界，使用控制点
                            spline_points = edge.control_points
                            points.extend([(p[0], p[1]) for p in spline_points])
                        else:
                            # 未支持的边类型
                            pass
                elif boundary_path.type == BoundaryPathType.POLYLINE:
                    # 多段线边界路径
                    vertices = boundary_path.vertices
                    points.extend(vertices)
                else:
                    # 未支持的边界路径类型
                    pass

        else:
            # 对于其他实体类型，尝试使用bbox方法
            bbox = entity.bbox()
            if bbox:
                (min_x, min_y, _), (max_x, max_y, _) = bbox.extmin, bbox.extmax
                return min_x, min_y, max_x, max_y
            else:
                return None

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            return min_x, min_y, max_x, max_y
        else:
            return None

    except Exception as e:
        # 异常处理，可以根据需要打印错误信息
        # print(f"Error processing entity {entity.dxf.handle} of type {entity_type}: {e}")
        return None

def check_overlap(bbox1, bbox2):
    """
    检查两个边界框是否重叠，按照指定的邻接定义。
    返回 True 表示重叠，False 表示不重叠。
    """
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    # 创建 shapely 的矩形对象
    rect1 = geometry.box(min_x1, min_y1, max_x1, max_y1)
    rect2 = geometry.box(min_x2, min_y2, max_x2, max_y2)

    # 检查是否重叠
    if rect1.overlaps(rect2):
        # 检查一个边框是否完全包含在另一个边框内
        if rect1.contains(rect2) or rect2.contains(rect1):
            return False
        else:
            return True
    else:
        return False

def get_feature_vectors(dxf_file_path):
    doc = ezdxf.readfile(dxf_file_path)
    msp = doc.modelspace()
    entities = []

    # 初始化每种实体类型的总数量
    total_counts = {etype: 0 for etype in ENTITY_TYPES}

    # 初始化每种实体类型的重叠实体集合
    overlapping_entities = {etype: set() for etype in ENTITY_TYPES}

    # 初始化每种实体类型与其他实体类型的重叠计数
    overlapping_counts = {
        etype: {other_type: 0 for other_type in ENTITY_TYPES if other_type != etype}
        for etype in ENTITY_TYPES
    }

    # 创建 R 树索引
    idx = index.Index()

    # 收集所有实体的信息，并插入到 R 树中
    for idx_id, entity in enumerate(msp):
        entity_type = entity.dxftype()
        if entity_type not in ENTITY_TYPES:
            continue
        bbox = compute_entity_bounding_box(entity, doc)
        if bbox:
            min_x, min_y, max_x, max_y = bbox
            entities.append({
                'type': entity_type,
                'bbox': bbox,
                'handle': entity.dxf.handle,
                'overlaps_with': set(),
                'id': idx_id  # 用于索引
            })
            total_counts[entity_type] += 1
            # 将实体的边界框插入到 R 树中，关联实体的索引 ID
            idx.insert(idx_id, (min_x, min_y, max_x, max_y))

    # 检查实体之间的重叠关系
    for entity in entities:
        etype = entity['type']
        bbox = entity['bbox']
        min_x, min_y, max_x, max_y = bbox
        entity_id = entity['id']

        # 使用 R 树查询可能与当前实体重叠的实体 ID 列表
        possible_matches = list(idx.intersection((min_x, min_y, max_x, max_y)))

        for match_id in possible_matches:
            if match_id == entity_id:
                continue  # 跳过自身

            other_entity = entities[match_id]
            other_type = other_entity['type']

            # 只考虑不同类型的实体
            if etype == other_type:
                continue

            # 检查是否已经处理过该对实体，避免重复计算
            if other_entity['id'] < entity_id:
                continue

            # 进行精确的重叠检查
            if check_overlap(bbox, other_entity['bbox']):
                # 记录重叠关系
                entity['overlaps_with'].add(other_type)
                other_entity['overlaps_with'].add(etype)

    # 统计重叠情况
    for entity in entities:
        etype = entity['type']
        overlaps = entity['overlaps_with']
        if overlaps:
            # 实体与其他实体类型发生了重叠
            overlapping_entities[etype].add(entity['handle'])
            for other_type in overlaps:
                overlapping_counts[etype][other_type] += 1  # 计数

    # 生成特征向量
    feature_vectors = {}
    for etype in ENTITY_TYPES:
        # 计算不与任何其他实体类型重叠的实体数量
        non_overlapping_count = total_counts[etype] - len(overlapping_entities[etype])

        # 构建特征向量
        vector = []
        for other_type in ENTITY_TYPES:
            if other_type == etype:
                vector.append(non_overlapping_count)
            else:
                vector.append(overlapping_counts[etype][other_type])
        feature_vectors[etype] = vector

    return feature_vectors

def normalize_feature_vectors(feature_vectors):
    """
    对特征矩阵的每一行进行行归一化
    """
    normalized_vectors = {}
    for etype, vector in feature_vectors.items():
        row_sum = sum(vector)
        if row_sum != 0:
            normalized_vector = [value / row_sum for value in vector]
        else:
            normalized_vector = vector  # 如果总和为0，保持原始值
        normalized_vectors[etype] = normalized_vector
    return normalized_vectors

def get_normalized_feature_matrix(dxf_file_path):
    """
    输入 DXF 文件路径，输出归一化的特征矩阵
    """
    feature_vectors = get_feature_vectors(dxf_file_path)
    normalized_vectors = normalize_feature_vectors(feature_vectors)
    # 将结果转换为二维列表，按照 ENTITY_TYPES 的顺序
    feature_matrix = [normalized_vectors[etype] for etype in ENTITY_TYPES]
    return feature_matrix

def print_feature_matrix(feature_matrix):
    """
    打印归一化后的特征矩阵
    """
    header = ENTITY_TYPES
    print('\t'.join(header))
    for i, vector in enumerate(feature_matrix):
        row = [str(round(value, 4)) for value in vector]
        print('\t'.join(row))

if __name__ == '__main__':
    dxf_file_path = r'C:\Users\15653\dwg-cx\dataset\modified\DFN6LCG(NiPdAu)（321）-517  Rev1_1.dxf'  # 请替换为您的DXF文件路径
    feature_matrix = get_normalized_feature_matrix(dxf_file_path)
    print("归一化的特征矩阵（每一行对应一个实体类型，列顺序为 ENTITY_TYPES）：")
    print_feature_matrix(feature_matrix)
