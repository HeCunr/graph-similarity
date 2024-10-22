import ezdxf
import math
from ezdxf.entities import BoundaryPathType, EdgeType

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
    返回True表示邻接，False表示不邻接。
    """
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    # 检查是否有重叠区域
    overlap_x = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
    overlap_y = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
    overlap_area = overlap_x * overlap_y

    # 如果重叠区域为零，或者边框仅在边界上接触，则不邻接
    if overlap_area == 0:
        return False

    # 检查一个边框是否完全包含在另一个边框内
    bbox1_area = (max_x1 - min_x1) * (max_y1 - min_y1)
    bbox2_area = (max_x2 - min_x2) * (max_y2 - min_y2)
    if (min_x1 >= min_x2 and max_x1 <= max_x2 and min_y1 >= min_y2 and max_y1 <= max_y2) or \
            (min_x2 >= min_x1 and max_x2 <= max_x1 and min_y2 >= min_y1 and max_y2 <= max_y1):
        return False

    # 有重叠区域，且不完全包含，则认为邻接
    return True

def get_adjacency_matrix(dxf_file_path):
    doc = ezdxf.readfile(dxf_file_path)
    msp = doc.modelspace()
    entities = []

    # 获取所有实体的边界框和类型
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type not in ENTITY_TYPES:
            continue
        bbox = compute_entity_bounding_box(entity, doc)
        if bbox:
            entities.append({
                'type': entity_type,
                'bbox': bbox,
                'handle': entity.dxf.handle
            })

    # 初始化邻接矩阵
    adjacency_matrix = {etype: {other_type: 0 for other_type in ENTITY_TYPES} for etype in ENTITY_TYPES}

    # 遍历所有实体，检查邻接关系
    for i in range(len(entities)):
        entity_i = entities[i]
        type_i = entity_i['type']
        bbox_i = entity_i['bbox']

        for j in range(i + 1, len(entities)):
            entity_j = entities[j]
            type_j = entity_j['type']
            bbox_j = entity_j['bbox']

            # 同类型实体不考虑
            if type_i == type_j:
                continue

            # 检查是否邻接
            if check_overlap(bbox_i, bbox_j):
                # 更新邻接计数
                adjacency_matrix[type_i][type_j] += 1
                adjacency_matrix[type_j][type_i] += 1  # 对称

    return adjacency_matrix

def print_adjacency_matrix(adjacency_matrix):
    # 打印邻接矩阵
    header = [''] + ENTITY_TYPES
    print('\t'.join(header))
    for etype in ENTITY_TYPES:
        row = [etype] + [str(adjacency_matrix[etype][other_type]) for other_type in ENTITY_TYPES]
        print('\t'.join(row))

if __name__ == '__main__':
    dxf_file_path = r'C:\Users\15653\dwg-cx\dataset\modified\DFN6LCG(NiPdAu)（321）-517  Rev1_1.dxf'  # 请替换为您的DXF文件路径
    adjacency_matrix = get_adjacency_matrix(dxf_file_path)
    print("邻接矩阵（行：实体类型，列：邻接的其他实体类型的数量）：")
    print_adjacency_matrix(adjacency_matrix)
