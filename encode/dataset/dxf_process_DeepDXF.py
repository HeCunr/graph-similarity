import os
import json
import numpy as np
import h5py
import ezdxf
import math

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
    elif entity_type == 'INSERT':
        insert_point = entity.dxf.insert
        points.append(insert_point)
    elif entity_type == 'SPLINE':
        # SPLINE 实体：获取控制点
        control_points = entity.control_points
        points.extend([(point[0], point[1]) for point in control_points])
    elif entity_type == 'SOLID':
        # SOLID 实体：获取顶点
        points.append((entity.dxf.vtx0.x, entity.dxf.vtx0.y))
        points.append((entity.dxf.vtx1.x, entity.dxf.vtx1.y))
        points.append((entity.dxf.vtx2.x, entity.dxf.vtx2.y))

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
    if max_val - min_val == 0:
        return 0
    return 2 * (value - min_val) / (max_val - min_val) - 1

def normalize_length(value, max_dim):
    """根据模型最大尺寸归一化长度参数"""
    if max_dim == 0:
        return 0
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
            features['vertices'] = normalized_vertices
            features['annotation_type'] = entity.dxf.annotation_type

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

        elif entity_type == 'SPLINE':
            # SPLINE 特征
            control_points = [(point[0], point[1]) for point in entity.control_points]
            normalized_control_points = [
                [
                    normalize_coordinate(p[0], min_x, max_x),
                    normalize_coordinate(p[1], min_y, max_y)
                ] for p in control_points
            ]
            knots = entity.knots
            avg_knots = sum(knots) / len(knots) if knots else 0  # 取knots的平均值
            features['control_points'] = normalized_control_points
            features['avg_knots'] = avg_knots

        elif entity_type == 'SOLID':
            # SOLID 特征
            points = [
                (entity.dxf.vtx0.x, entity.dxf.vtx0.y),
                (entity.dxf.vtx1.x, entity.dxf.vtx1.y),
                (entity.dxf.vtx2.x, entity.dxf.vtx2.y)
            ]
            normalized_points = [
                [
                    normalize_coordinate(p[0], min_x, max_x),
                    normalize_coordinate(p[1], min_y, max_y)
                ] for p in points
            ]
            features['points'] = normalized_points

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

    return output_json_path  # 返回生成的 JSON 文件路径

def process_entity(entity):
    entity_type = entity['type']
    features = entity['features']
    new_features = {}

    if entity_type == 'HATCH':
        # HATCH 特征
        new_features['solid_fill'] = features.get('solid_fill', 0)
        new_features['associative'] = features.get('associative', 0)
        new_features['boundary_paths'] = features.get('boundary_paths', 0)

    elif entity_type == 'TEXT':
        # TEXT 特征
        point = features.get('insert_point', [0, 0])
        new_features['text_insert_point_x'] = point[0]
        new_features['text_insert_point_y'] = point[1]
        new_features['height'] = features.get('height', 0)
        new_features['text_rotation'] = features.get('rotation', 0)

    elif entity_type == 'MTEXT':
        # MTEXT 特征
        point = features.get('insert_point', [0, 0])
        new_features['mtext_insert_point_x'] = point[0]
        new_features['mtext_insert_point_y'] = point[1]
        new_features['char_height'] = features.get('char_height', 0)
        new_features['width'] = features.get('width', 0)

    elif entity_type == 'LWPOLYLINE':
        # LWPOLYLINE 特征
        points = features.get('points', [])
        if points:
            x_sum = sum(p[0] for p in points)
            y_sum = sum(p[1] for p in points)
            avg_x = x_sum / len(points)
            avg_y = y_sum / len(points)
        else:
            avg_x = avg_y = 0
        new_features['closed'] = features.get('closed', False)
        new_features['points_x'] = avg_x
        new_features['points_y'] = avg_y
        new_features['count'] = features.get('count', 0)

    elif entity_type == 'ARC':
        # ARC 特征
        point = features.get('center', [0, 0])
        new_features['arc_center_x'] = point[0]
        new_features['arc_center_y'] = point[1]
        new_features['arc_radius'] = features.get('radius', 0)
        new_features['start_angle'] = features.get('start_angle', 0)
        new_features['end_angle'] = features.get('end_angle', 0)

    elif entity_type == 'LINE':
        # LINE 特征
        start_point = features.get('start_point', [0, 0])
        end_point = features.get('end_point', [0, 0])
        new_features['start_point_x'] = start_point[0]
        new_features['start_point_y'] = start_point[1]
        new_features['end_point_x'] = end_point[0]
        new_features['end_point_y'] = end_point[1]

    elif entity_type == 'CIRCLE':
        # CIRCLE 特征
        point = features.get('center', [0, 0])
        new_features['circle_center_x'] = point[0]
        new_features['circle_center_y'] = point[1]
        new_features['circle_radius'] = features.get('radius', 0)

    elif entity_type == 'DIMENSION':
        # DIMENSION 特征
        defpoint = features.get('defpoint', [0, 0])
        text_midpoint = features.get('text_midpoint', [0, 0])
        new_features['defpoint_x'] = defpoint[0]
        new_features['defpoint_y'] = defpoint[1]
        new_features['text_midpoint_x'] = text_midpoint[0]
        new_features['text_midpoint_y'] = text_midpoint[1]
        new_features['dim_type'] = features.get('dim_type', 0)

    elif entity_type == 'LEADER':
        # LEADER 特征
        vertices = features.get('vertices', [])
        if vertices:
            x_sum = sum(v[0] for v in vertices)
            y_sum = sum(v[1] for v in vertices)
            avg_x = x_sum / len(vertices)
            avg_y = y_sum / len(vertices)
        else:
            avg_x = avg_y = 0
        new_features['vertices_x'] = avg_x
        new_features['vertices_y'] = avg_y
        new_features['annotation_type'] = features.get('annotation_type', 0)

    elif entity_type == 'INSERT':
        # INSERT 特征
        point = features.get('insert_point', [0, 0])
        new_features['insert_insert_point_x'] = point[0]
        new_features['insert_insert_point_y'] = point[1]
        scale = features.get('scale', (1, 1))
        new_features['scale_x'] = scale[0]
        new_features['scale_y'] = scale[1]
        new_features['insert_rotation'] = features.get('rotation', 0)

    elif entity_type == 'SPLINE':
        # SPLINE 特征
        control_points = features.get('control_points', [])
        if control_points:
            x_sum = sum(p[0] for p in control_points)
            y_sum = sum(p[1] for p in control_points)
            avg_x = x_sum / len(control_points)
            avg_y = y_sum / len(control_points)
        else:
            avg_x = avg_y = 0
        new_features['control_points_x'] = avg_x
        new_features['control_points_y'] = avg_y
        new_features['avg_knots'] = features.get('avg_knots', 0)

    elif entity_type == 'SOLID':
        # SOLID 特征
        points = features.get('points', [])
        if points:
            x_sum = sum(p[0] for p in points)
            y_sum = sum(p[1] for p in points)
            avg_x = x_sum / len(points)
            avg_y = y_sum / len(points)
        else:
            avg_x = avg_y = 0
        new_features['solid_points_x'] = avg_x
        new_features['solid_points_y'] = avg_y

    else:
        # 其他未定义的实体类型，保持原样或根据需要处理
        new_features = features

    return {
        'type': entity_type,
        'features': new_features
    }

def process_json_file(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_entities = []
    for entity in data:
        processed_entity = process_entity(entity)
        processed_entities.append(processed_entity)

    with open(output_path, 'w') as f:
        json.dump(processed_entities, f, indent=2)

    print(f"已处理文件 {input_path}，结果保存至 {output_path}")

    return output_path  # 返回处理后的 JSON 文件路径

class DXFEntity:
    def __init__(self, entity_type, features):
        self.type = entity_type
        self.features = features

    @staticmethod
    def from_dict(data):
        entity_type = data['type']
        features = data['features']
        if entity_type == 'LINE':
            return Line.from_dict(features)
        elif entity_type == 'CIRCLE':
            return Circle.from_dict(features)
        elif entity_type == 'ARC':
            return Arc.from_dict(features)
        elif entity_type == 'LWPOLYLINE':
            return LWPolyline.from_dict(features)
        elif entity_type == 'TEXT':
            return Text.from_dict(features)
        elif entity_type == 'MTEXT':
            return MText.from_dict(features)
        elif entity_type == 'HATCH':
            return Hatch.from_dict(features)
        elif entity_type == 'DIMENSION':
            return Dimension.from_dict(features)
        elif entity_type == 'LEADER':
            return Leader.from_dict(features)
        elif entity_type == 'INSERT':
            return Insert.from_dict(features)
        elif entity_type == 'SPLINE':  # 添加对SPLINE的处理
            return Spline.from_dict(features)
        elif entity_type == 'SOLID':  # 添加对SOLID的处理
            return Solid.from_dict(features)
        elif entity_type == 'EOS':
            return DXFEntity('EOS', {})
        else:
            return DXFEntity(entity_type, features)

    def to_vector(self):
        vector = [-1] * (len(FEATURE_NAMES) + 1)  # +1 for entity type
        vector[0] = ENTITY_TYPES.index(self.type) if self.type in ENTITY_TYPES else -1
        for i, feat in enumerate(FEATURE_NAMES):
            if feat in self.features:
                vector[i+1] = self.features[feat]
        return vector

    def quantize(self, value, is_angle=False):
        if is_angle:
            value = value / 360.0  # Normalize angle
        return int(np.clip(np.round((value + 1.0) / 2 * 255), 0, 255))  # Convert to int

class Line(DXFEntity):
    def __init__(self, start_point_x, start_point_y, end_point_x, end_point_y):
        features = {
            'start_point_x': start_point_x,
            'start_point_y': start_point_y,
            'end_point_x': end_point_x,
            'end_point_y': end_point_y
        }
        super().__init__('LINE', features)

    @staticmethod
    def from_dict(features):
        return Line(features['start_point_x'], features['start_point_y'],
                    features['end_point_x'], features['end_point_y'])

    def normalize(self):
        for key in ['start_point_x', 'start_point_y', 'end_point_x', 'end_point_y']:
            self.features[key] = self.quantize(self.features[key])

class Circle(DXFEntity):
    def __init__(self, circle_center_x, circle_center_y, circle_radius):
        features = {
            'circle_center_x': circle_center_x,
            'circle_center_y': circle_center_y,
            'circle_radius': circle_radius
        }
        super().__init__('CIRCLE', features)

    @staticmethod
    def from_dict(features):
        return Circle(features['circle_center_x'], features['circle_center_y'],
                      features['circle_radius'])

    def normalize(self):
        for key in ['circle_center_x', 'circle_center_y', 'circle_radius']:
            self.features[key] = self.quantize(self.features[key])

class Arc(DXFEntity):
    def __init__(self, arc_center_x, arc_center_y, arc_radius, start_angle, end_angle):
        features = {
            'arc_center_x': arc_center_x,
            'arc_center_y': arc_center_y,
            'arc_radius': arc_radius,
            'start_angle': start_angle,
            'end_angle': end_angle
        }
        super().__init__('ARC', features)

    @staticmethod
    def from_dict(features):
        return Arc(features['arc_center_x'], features['arc_center_y'],
                   features['arc_radius'], features['start_angle'],
                   features['end_angle'])

    def normalize(self):
        for key in ['arc_center_x', 'arc_center_y', 'arc_radius']:
            self.features[key] = self.quantize(self.features[key])
        for key in ['start_angle', 'end_angle']:
            self.features[key] = self.quantize(self.features[key], is_angle=True)

class LWPolyline(DXFEntity):
    def __init__(self, closed, points_x, points_y, count):
        features = {
            'closed': closed,
            'points_x': points_x,
            'points_y': points_y,
            'count': count
        }
        super().__init__('LWPOLYLINE', features)

    @staticmethod
    def from_dict(features):
        return LWPolyline(features['closed'], features['points_x'],
                          features['points_y'], features['count'])

    def normalize(self):
        self.features['closed'] = int(self.features['closed'])
        for key in ['points_x', 'points_y']:
            self.features[key] = self.quantize(self.features[key])
        self.features['count'] = int(np.clip(self.features['count'], 0, 255))

class Text(DXFEntity):
    def __init__(self, text_insert_point_x, text_insert_point_y, height, text_rotation):
        features = {
            'text_insert_point_x': text_insert_point_x,
            'text_insert_point_y': text_insert_point_y,
            'height': height,
            'text_rotation': text_rotation
        }
        super().__init__('TEXT', features)

    @staticmethod
    def from_dict(features):
        return Text(features['text_insert_point_x'], features['text_insert_point_y'],
                    features['height'], features['text_rotation'])

    def normalize(self):
        for key in ['text_insert_point_x', 'text_insert_point_y', 'height']:
            self.features[key] = self.quantize(self.features[key])
        self.features['text_rotation'] = self.quantize(self.features['text_rotation'], is_angle=True)

class MText(DXFEntity):
    def __init__(self, mtext_insert_point_x, mtext_insert_point_y, char_height, width):
        features = {
            'mtext_insert_point_x': mtext_insert_point_x,
            'mtext_insert_point_y': mtext_insert_point_y,
            'char_height': char_height,
            'width': width
        }
        super().__init__('MTEXT', features)

    @staticmethod
    def from_dict(features):
        return MText(features['mtext_insert_point_x'], features['mtext_insert_point_y'],
                     features['char_height'], features['width'])

    def normalize(self):
        for key in ['mtext_insert_point_x', 'mtext_insert_point_y', 'char_height', 'width']:
            self.features[key] = self.quantize(self.features[key])

class Hatch(DXFEntity):
    def __init__(self, solid_fill, associative, boundary_paths):
        features = {
            'solid_fill': solid_fill,
            'associative': associative,
            'boundary_paths': boundary_paths
        }
        super().__init__('HATCH', features)

    @staticmethod
    def from_dict(features):
        return Hatch(features['solid_fill'], features['associative'],
                     features['boundary_paths'])

    def normalize(self):
        self.features['solid_fill'] = int(self.features['solid_fill'])
        self.features['associative'] = int(self.features['associative'])
        self.features['boundary_paths'] = int(np.clip(self.features['boundary_paths'], 0, 255))

class Dimension(DXFEntity):
    def __init__(self, defpoint_x, defpoint_y, text_midpoint_x, text_midpoint_y):
        features = {
            'defpoint_x': defpoint_x,
            'defpoint_y': defpoint_y,
            'text_midpoint_x': text_midpoint_x,
            'text_midpoint_y': text_midpoint_y
        }
        super().__init__('DIMENSION', features)

    @staticmethod
    def from_dict(features):
        return Dimension(features['defpoint_x'], features['defpoint_y'],
                         features['text_midpoint_x'], features['text_midpoint_y'])

    def normalize(self):
        for key in ['defpoint_x', 'defpoint_y', 'text_midpoint_x', 'text_midpoint_y']:
            self.features[key] = self.quantize(self.features[key])

class Leader(DXFEntity):
    def __init__(self, vertices_x, vertices_y):
        features = {
            'vertices_x': vertices_x,
            'vertices_y': vertices_y
        }
        super().__init__('LEADER', features)

    @staticmethod
    def from_dict(features):
        return Leader(features['vertices_x'], features['vertices_y'])

    def normalize(self):
        for key in ['vertices_x', 'vertices_y']:
            self.features[key] = self.quantize(self.features[key])

class Insert(DXFEntity):
    def __init__(self, insert_insert_point_x, insert_insert_point_y, scale_x, scale_y, insert_rotation):
        features = {
            'insert_insert_point_x': insert_insert_point_x,
            'insert_insert_point_y': insert_insert_point_y,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'insert_rotation': insert_rotation
        }
        super().__init__('INSERT', features)

    @staticmethod
    def from_dict(features):
        return Insert(features['insert_insert_point_x'], features['insert_insert_point_y'],
                      features['scale_x'], features['scale_y'], features['insert_rotation'])

    def normalize(self):
        for key in ['insert_insert_point_x', 'insert_insert_point_y', 'scale_x', 'scale_y']:
            self.features[key] = self.quantize(self.features[key])
        self.features['insert_rotation'] = self.quantize(self.features['insert_rotation'], is_angle=True)


class Spline(DXFEntity):
    def __init__(self, control_points_x, control_points_y, avg_knots):
        features = {
            'control_points_x': control_points_x,
            'control_points_y': control_points_y,
            'avg_knots': avg_knots
        }
        super().__init__('SPLINE', features)

    @staticmethod
    def from_dict(features):
        return Spline(
            features.get('control_points_x', 0),
            features.get('control_points_y', 0),
            features.get('avg_knots', 0)
        )

    def normalize(self):
        for key in ['control_points_x', 'control_points_y']:
            self.features[key] = self.quantize(self.features[key])
        self.features['avg_knots'] = self.quantize(self.features['avg_knots'])


class Solid(DXFEntity):
    def __init__(self, solid_points_x, solid_points_y):
        features = {
            'solid_points_x': solid_points_x,
            'solid_points_y': solid_points_y
        }
        super().__init__('SOLID', features)

    @staticmethod
    def from_dict(features):
        return Solid(
            features.get('solid_points_x', 0),
            features.get('solid_points_y', 0)
        )

    def normalize(self):
        for key in ['solid_points_x', 'solid_points_y']:
            self.features[key] = self.quantize(self.features[key])

class DXFSequence:
    def __init__(self, entities):
        self.entities = entities

    @staticmethod
    def from_dict(json_data):
        entities = [DXFEntity.from_dict(entity) for entity in json_data]
        return DXFSequence(entities)

    def normalize(self):
        for entity in self.entities:
            if entity.type != 'EOS':
                entity.normalize()

    @staticmethod
    def eos_vector():
        vector = [-1] * (len(FEATURE_NAMES) + 1)
        vector[0] = ENTITY_TYPES.index('EOS')
        return vector

    def to_vector(self, max_len=4096):
        vectors = []
        for entity in self.entities:
            vector = entity.to_vector()
            vectors.append(vector)
        seq_len = len(vectors)
        if seq_len > max_len:
            return None
        eos_vec = self.eos_vector()
        vectors += [eos_vec] * (max_len - seq_len)
        return np.array(vectors, dtype=np.int16)

ENTITY_TYPES = ['LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT',
                'MTEXT', 'HATCH', 'DIMENSION', 'LEADER', 'INSERT', 'SPLINE', 'SOLID', 'EOS']

FEATURE_NAMES = [
    'solid_fill', 'associative', 'boundary_paths',
    'text_insert_point_x', 'text_insert_point_y', 'height', 'text_rotation',
    'mtext_insert_point_x', 'mtext_insert_point_y', 'char_height', 'width',
    'closed', 'points_x', 'points_y', 'count',
    'arc_center_x', 'arc_center_y', 'arc_radius', 'start_angle', 'end_angle',
    'start_point_x', 'start_point_y', 'end_point_x', 'end_point_y',
    'circle_center_x', 'circle_center_y', 'circle_radius',
    'defpoint_x', 'defpoint_y', 'text_midpoint_x', 'text_midpoint_y',
    'vertices_x', 'vertices_y',
    'insert_insert_point_x', 'insert_insert_point_y', 'scale_x', 'scale_y',
    'insert_rotation', 'control_points_x', 'control_points_y', 'avg_knots',
    'solid_points_x', 'solid_points_y'
]

def load_dxf(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    dxf_seq = DXFSequence.from_dict(json_data)
    dxf_seq.normalize()
    vector = dxf_seq.to_vector()
    return vector



def process_dxf_files_for_deepdxf(input_dir, output_h5_dir):
    if not os.path.exists(input_dir):
        print(f"输入目录 {input_dir} 不存在。")
        return

    if not os.path.exists(output_h5_dir):
        os.makedirs(output_h5_dir)

    temp_dir = os.path.join(os.getcwd(), '../temp')
    os.makedirs(temp_dir, exist_ok=True)

    raw_json_dir = os.path.join(temp_dir, 'raw_json')
    os.makedirs(raw_json_dir, exist_ok=True)

    processed_json_dir = os.path.join(temp_dir, 'processed_json')
    os.makedirs(processed_json_dir, exist_ok=True)

    # 获取输入目录下的所有 DXF 文件
    dxf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.dxf')]

    if not dxf_files:
        print(f"在目录 {input_dir} 中未找到任何 DXF 文件。")
        return

    for dxf_file in dxf_files:
        input_dxf_path = os.path.join(input_dir, dxf_file)
        print(f"正在处理文件: {input_dxf_path}")

        # Step 1: 处理 DXF 文件，生成初始 JSON 文件
        raw_json_path = process_dxf_file(input_dxf_path, raw_json_dir)

        # Step 2: 处理初始 JSON 文件，生成处理后的 JSON 文件
        input_json_path = raw_json_path
        output_json_filename = os.path.basename(raw_json_path)
        processed_json_path = os.path.join(processed_json_dir, output_json_filename)
        process_json_file(input_json_path, processed_json_path)

        # Step 3: 将处理后的 JSON 文件转换为向量
        dxf_vec = load_dxf(processed_json_path)

        if dxf_vec is None:
            print(f"文件 {dxf_file} 的向量表示超过了最大长度，跳过此文件。")
            continue

        # Step 4: 将向量保存到 H5 文件
        h5_filename = os.path.splitext(dxf_file)[0] + '.h5'
        h5_path = os.path.join(output_h5_dir, h5_filename)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('dxf_vec', data=np.array([dxf_vec]), dtype=np.int16)

        print(f"已处理 {input_dxf_path} 并保存到 {h5_path}")

    print("所有文件处理完成。")

if __name__ == '__main__':
    # 示例用法
    input_dir = r'/home/vllm/encode/data/DeepDXF/TEST'  # 修改为您的 DXF 文件目录
    output_h5_dir =  r'/home/vllm/encode/data/DeepDXF/TEST_4096'   # 修改为您希望保存 H5 文件的目录
    process_dxf_files_for_deepdxf(input_dir, output_h5_dir)

