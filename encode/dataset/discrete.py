import ezdxf
import math
import os
import re
import json
import numpy as np
from ezdxf.entities import BoundaryPathType, EdgeType

# 定义实体类型列表
ENTITY_TYPES = ['LINE', 'SPLINE', 'CIRCLE', 'ARC', 'ELLIPSE', 'MTEXT', 'LEADER', 'HATCH', 'DIMENSION', 'SOLID' ]

def process_coordinate(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0

    value = 2 * (value - min_val) / (max_val - min_val) - 1
    value = np.array([value])
    value = ((value + 1.0) / 2 * 256).round().clip(min=0, max=255).astype(np.int)[0]
    return value

def process_length(value, max_dim):
    if max_dim == 0:
        return 0

    value = 2 * value / max_dim - 1
    value = np.array([value])
    value = ((value + 1.0) / 2 * 256).round().clip(min=0, max=255).astype(np.int)[0]
    return value

def get_entity_points(entity):
    """获取实体的所有坐标点，用于计算边界框"""
    entity_type = entity.dxftype()
    points = []

    if entity_type == 'LINE':
        points.append(entity.dxf.start)
        points.append(entity.dxf.end)
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
    elif entity_type in ['MTEXT']:
        insert_point = entity.dxf.insert
        points.append(insert_point)
    elif entity_type == 'HATCH':
        for path in entity.paths:
            if hasattr(path, 'edges'):  # EdgePath类型
                for edge in path.edges:
                    if hasattr(edge, 'start') and hasattr(edge, 'end'):
                        points.append(edge.start)
                        points.append(edge.end)
                    elif hasattr(edge, 'center') and hasattr(edge, 'radius'):
                        # 处理圆弧边界
                        center = edge.center
                        radius = edge.radius
                        start_angle = getattr(edge, 'start_angle', 0)
                        end_angle = getattr(edge, 'end_angle', 360)
                        angles = np.linspace(start_angle, end_angle, 36)  # 每10度采样一个点
                        points.extend([
                            (center[0] + radius * np.cos(np.radians(angle)),
                             center[1] + radius * np.sin(np.radians(angle)))
                            for angle in angles
                        ])
            elif hasattr(path, 'vertices'):  # PolylinePath类型
                points.extend(path.vertices)
    elif entity_type == 'DIMENSION':
        defpoint = entity.dxf.defpoint
        text_midpoint = entity.dxf.text_midpoint
        points.append(defpoint)
        points.append(text_midpoint)
    elif entity_type == 'LEADER':
        points.extend(entity.vertices)
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

def parse_text(text):
    # 第一种格式：0.038
    match1 = re.match(r'^(\d+\.\d+)$', text)
    if match1:
        text_0 = match1.group(1)
        text_1 = "0"
        return float(text_0), int(text_1)

    # 第二种格式：0.200%%P0.025
    match2 = re.search(r'(\d+\.\d+)?%%[Pp]?(\d+\.\d+)?', text)
    if match2:
        text_0 = match2.group(1) if match2.group(1) else "0"
        text_1 = match2.group(2) if match2.group(2) else "0"
        return float(text_0), float(text_1)

    return 0, 0

def extract_line_dim(dim, doc):
    block = doc.blocks.get(dim.dxf.geometry)
    if block is not None:
        for entity in block:
            if entity.dxftype() in ['TEXT', 'MTEXT']:
                text = entity.dxf.text
                value_0 = extract_numbers(text)
                return value_0

def extract_numbers(text):
    pattern = r'\\A1;(\d+\.\d+)'
    match = re.search(pattern, text)
    if match:
        number = match.group(1) if match.group(1) else "0"
        return float(number)
    else:
        return 0

def get_entity_info(entity, doc, min_x, min_y, max_x, max_y, max_dim):
    entity_type = entity.dxftype()

    # 定义一个空列表存储实体信息
    entity_info = []

    if entity_type == 'LINE':
        start_point = entity.dxf.start
        end_point = entity.dxf.end
        # 确保编码唯一性
        if start_point[0] > end_point[0]:
            temp = start_point
            start_point = end_point
            end_point = temp

        elif (start_point[0] == end_point[0]) and (start_point[1] > end_point[1]):
            temp = start_point
            start_point = end_point
            end_point = temp

        value_0 = process_coordinate(start_point[0], min_x, max_x)
        value_1 = process_coordinate(start_point[1], min_y, max_y)
        value_2 = process_coordinate(end_point[0], min_x, max_x)
        value_3 = process_coordinate(end_point[1], min_y, max_y)

        entity_info = [value_0, value_1, value_2, value_3]

    elif entity_type == 'SPLINE':
        vertices = entity.fit_points
        if len(vertices) >= 2:
            value_0 = process_coordinate(vertices[0][0], min_x, max_x)
            value_1 = process_coordinate(vertices[1][0], min_x, max_x)
            value_2 = process_coordinate(vertices[-1][0], min_x, max_x)
            value_3 = process_coordinate(vertices[0][1], min_y, max_y)
            value_4 = process_coordinate(vertices[1][1], min_y, max_y)
            value_5 = process_coordinate(vertices[-1][1], min_y, max_y)
            entity_info = [value_0, value_1, value_2, value_3, value_4, value_5]

    elif entity_type == 'CIRCLE':
        center = entity.dxf.center
        radius = entity.dxf.radius

        value_0 = process_coordinate(center[0], min_x, max_x)
        value_1 = process_coordinate(center[1], min_y, max_y)
        value_2 = process_length(radius, max_dim)
        entity_info = [value_0, value_1, value_2]

    elif entity_type == 'ARC':
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)

        value_0 = process_coordinate(center[0], min_x, max_x)
        value_1 = process_coordinate(center[1], min_y, max_y)
        value_2 = process_length(radius, max_dim)
        value_3 = process_coordinate(start_angle, -360, 360)
        value_4 = process_coordinate(end_angle, -360, 360)
        entity_info = [value_0, value_1, value_2, value_3, value_4]

    elif entity_type == 'ELLIPSE':
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio

        value_0 = process_coordinate(center[0], min_x, max_x)
        value_1 = process_coordinate(center[1], min_y, max_y)
        value_2 = process_coordinate(major_axis[0], min_x, max_x)
        value_3 = process_coordinate(major_axis[1], min_y, max_y)
        value_4 = process_coordinate(ratio, 0, 1)
        entity_info = [value_0, value_1, value_2, value_3, value_4]

    elif entity_type == 'MTEXT':
        insert = entity.dxf.insert
        height = entity.dxf.char_height

        value_0 = process_coordinate(insert[0], min_x, max_x)
        value_1 = process_coordinate(insert[1], min_y, max_y)
        value_2 = process_length(height, max_dim)

        entity_info = [value_0, value_1,  value_2]

    elif entity_type == 'HATCH':
        bbox = compute_entity_bounding_box(entity)
        value_0 = process_coordinate(bbox[0], min_x, max_x)
        value_1 = process_coordinate(bbox[2], min_x, max_x)
        value_2 = process_coordinate(bbox[1], min_y, max_y)
        value_3 = process_coordinate(bbox[3], min_y, max_y)
        entity_info = [value_0, value_1, value_2, value_3]

    elif entity_type == 'DIMENSION':
        def_point = entity.dxf.defpoint
        midpoint = entity.dxf.text_midpoint
        text = entity.dxf.text

        value_4, value_5 = parse_text(text)
        if value_4 == 0:
            value_4 = extract_line_dim(entity, doc)

        value_0 = process_coordinate(def_point[0], min_x, max_x)
        value_1 = process_coordinate(midpoint[0], min_x, max_x)
        value_2 = process_coordinate(def_point[1], min_y, max_y)
        value_3 = process_coordinate(midpoint[1], min_y, max_y)

        value_4 = value_4 * 1000
        value_5 = value_5 * 1000

        entity_info = [value_0, value_1, value_2,value_3,value_4,value_5]

    elif entity_type == 'LEADER':
        vertices = entity.vertices
        value_0 = process_coordinate(vertices[0][0], min_x, max_x)
        value_1 = process_coordinate(vertices[-1][0], min_x, max_x)
        value_2 = process_coordinate(vertices[0][1], min_y, max_y)
        value_3 = process_coordinate(vertices[-1][1], min_y, max_y)
        entity_info = [value_0, value_1, value_2,value_3]

    elif entity_type == 'SOLID':
        vertices = entity.vertices()
        if len(vertices) >= 2:
            value_0 = process_coordinate(vertices[0][0], min_x, max_x)
            value_1 = process_coordinate(vertices[1][0], min_x, max_x)
            value_2 = process_coordinate(vertices[2][0], min_x, max_x)
            value_3 = process_coordinate(vertices[0][1], min_y, max_y)
            value_4 = process_coordinate(vertices[1][1], min_y, max_y)
            value_5 = process_coordinate(vertices[2][1], min_y, max_y)
            entity_info = [value_0, value_1, value_2, value_3, value_4, value_5]

    return entity_info

def compute_entity_bounding_box(entity):
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

        elif entity_type == 'ELLIPSE':
            # 椭圆边界，采样椭圆上的点
            center = entity.dxf.center
            major_axis = entity.dxf.major_axis
            ratio = entity.dxf.ratio
            start_param = entity.dxf.start_param
            end_param = entity.dxf.end_param
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

        elif entity_type == 'MTEXT':
            # 使用文本的插入点，假设文本是水平的，使用高度和宽度
            insert_point = entity.dxf.insert
            height = entity.dxf.char_height
            width = entity.dxf.width if hasattr(entity.dxf, 'width') else 0
            x, y = insert_point[0], insert_point[1]
            min_x = x
            max_x = x + width
            min_y = y
            max_y = y + height
            return min_x, min_y, max_x, max_y

        elif entity_type == 'SPLINE':
            points = entity.fit_points

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
                            ratio = edge.ratio
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

        elif entity_type == 'SOLID':
            vertices = entity.vertices()
            p1 = vertices[0]
            p2 = vertices[1]
            p3 = vertices[2]

            # 计算三角形外接正方形的四个顶点
            squ_vertices = bounding_square_of_triangle(p1, p2, p3)
            min_x = squ_vertices[2][0]
            min_y = squ_vertices[2][1]
            max_x = squ_vertices[1][0]
            max_y = squ_vertices[1][1]
            return min_x, min_y, max_x, max_y

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
        return None

def bounding_square_of_triangle(p1, p2, p3):
    min_x = min(p1[0], p2[0], p3[0])
    max_x = max(p1[0], p2[0], p3[0])
    min_y = min(p1[1], p2[1], p3[1])
    max_y = max(p1[1], p2[1], p3[1])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    side_length = max(max_x - min_x, max_y - min_y)

    top_left = (center_x - side_length/2, center_y + side_length/2)
    top_right = (center_x + side_length/2, center_y + side_length/2)
    bottom_left = (center_x - side_length/2, center_y - side_length/2)
    bottom_right = (center_x + side_length/2, center_y - side_length/2)

    return top_left, top_right, bottom_left, bottom_right

def check_overlap(bbox1, bbox2):
    """
    检查两个边界框是否重叠，根据新的邻接定义。
    返回True表示邻接，False表示不邻接。
    """
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    # 检查是否有重叠区域
    overlap_x =  min(max_x1, max_x2) - max(min_x1, min_x2)
    overlap_y =  min(max_y1, max_y2) - max(min_y1, min_y2)

    # 如果有任何重叠（包括边框边界重叠和顶点重合），则认为邻接
    if overlap_x >= 0 and overlap_y >= 0:
        return True

    return False

def get_adjacency_list(adj_matrix):
    adj_list = []
    for row in adj_matrix:
        adj_list.append([i for i, val in enumerate(row) if val == 1])

    return adj_list

def get_adj_feat_matrix(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    min_x, min_y, max_x, max_y = get_bounding_box(doc)
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)

    # 设置单一类型实体单张图最大节点数, 节点属性维度
    batch_size = 200
    d = 6

    # 按entity_types顺序排列实体类型,按实体边界框中心点顺序排列某类型实体
    entities = {key: [] for key in ENTITY_TYPES}
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type not in ENTITY_TYPES:
            continue

        info = get_entity_info(entity, doc, min_x, min_y, max_x, max_y, max_dim)
        bbox = compute_entity_bounding_box(entity)
        if bbox is not None:
            p = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            entities[entity_type].append({
                'coordinates': p,
                'bbox': bbox,
                'entity_info': info
            })

    # 用字典存储按类型划分的邻接表,属性矩阵
    adj_lists = {key: [] for key in ENTITY_TYPES}
    feat_lists = {key: [] for key in ENTITY_TYPES}

    for entity_type in ENTITY_TYPES:
        sorted_entities = sorted(entities[entity_type], key=lambda x: (x['coordinates'][0], -x['coordinates'][1]))
        num_groups = len(sorted_entities) // batch_size

        for i in range(num_groups):
            # 初始化邻接矩阵，属性矩阵
            adj_matrix = np.zeros((batch_size, batch_size), dtype=int)
            adj_matrix.fill(-1)
            feat_matrix = np.zeros((batch_size, d), dtype=float)
            feat_matrix.fill(-1)
            for j in range(i*batch_size, (i+1)*batch_size):
                entity_j = sorted_entities[j]
                bbox_j = entity_j['bbox']
                entity_info_j = entity_j['entity_info']
                for k in range(j+1, (i+1)*batch_size):
                    entity_k = sorted_entities[k]
                    bbox_k = entity_k['bbox']

                    # 检查是否邻接
                    if check_overlap(bbox_j, bbox_k):
                        # 更新邻接计数
                        adj_matrix[(j%batch_size)][(k%batch_size)] = 1
                        adj_matrix[(k%batch_size)][(j%batch_size)] = 1  # 对称

                for s in range(len(entity_info_j)):
                    feat_matrix[(j%batch_size)][s] = entity_info_j[s]

            # 转化为列表
            adj_matrix = np.triu(adj_matrix, k=0)
            adj_list = get_adjacency_list(adj_matrix)
            feat_matrix = np.round(feat_matrix, 3)
            feat_list = [[elem for elem in row] for row in feat_matrix]

            adj_lists[entity_type].append(adj_list)
            feat_lists[entity_type].append(feat_list)

        if len(sorted_entities) % batch_size > 1:
            last_batch = sorted_entities[num_groups * batch_size:]
            batch = len(sorted_entities) % batch_size
            adj_matrix = np.zeros((batch, batch), dtype=int)
            adj_matrix.fill(-1)
            feat_matrix = np.zeros((batch, d), dtype=float)
            feat_matrix.fill(-1)
            for i in range(batch):
                entity_i = last_batch[i]
                bbox_i = entity_i['bbox']
                entity_info_i = entity_i['entity_info']
                for j in range(i+1, batch):
                    entity_j = last_batch[j]
                    bbox_j = entity_j['bbox']

                    # 检查是否邻接
                    if check_overlap(bbox_i, bbox_j):
                        # 更新邻接计数
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1  # 对称

                for s in range(len(entity_info_i)):
                    feat_matrix[i][s] = entity_info_i[s]

            # 转化为列表
            adj_matrix = np.triu(adj_matrix, k=0)
            adj_list = get_adjacency_list(adj_matrix)
            feat_matrix = np.round(feat_matrix, 3)
            feat_list = [[elem for elem in row] for row in feat_matrix]

            adj_lists[entity_type].append(adj_list)
            feat_lists[entity_type].append(feat_list)

    return adj_lists, feat_lists

def dataset_generate(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            file_path = os.path.join(folder_path, filename)
            file_name = os.path.splitext(filename)[0]
            json_name = os.path.join('/mnt/csip-108/cl/lf_dra_mat/CGMN/data/TR', f"{file_name}.json")

            # 获取邻接矩阵和属性矩阵
            adj_lists, feat_lists = get_adj_feat_matrix(file_path)

            h = 0
            with open(json_name, 'w') as json_file:
                for entity_type in ENTITY_TYPES:
                    for i in range(len(adj_lists[entity_type])):
                        num_nodes = len(adj_lists[entity_type][i])
                        file_data = {
                            "src": entity_type,
                            "n_num": num_nodes,
                            "succs": adj_lists[entity_type][i],
                            "features": feat_lists[entity_type][i],
                            "fname": file_path + str(h)
                        }
                        h += 1

                        # 将文件数据写入 JSON 文件
                        json.dump(file_data, json_file)
                        json_file.write('\n')


if __name__ == '__main__':
    dxf_folder_path = '/mnt/csip-108/cl/lf_dra_mat/second/explode_dataset'  # 请替换为您的DXF文件路径
    dataset_generate(dxf_folder_path)


