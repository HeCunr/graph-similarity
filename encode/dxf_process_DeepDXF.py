import ezdxf
import math
import json
import numpy as np

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
    if max_val - min_val == 0:
        return 0
    return 2 * (value - min_val) / (max_val - min_val) - 1

def normalize_length(value, max_dim):
    """根据模型最大尺寸归一化长度参数"""
    if max_dim == 0:
        return 0
    return value * (2 / max_dim)

def process_dxf_file(input_dxf_path):
    """处理单个 DXF 文件，返回处理后的实体列表"""
    doc = ezdxf.readfile(input_dxf_path)
    msp = doc.modelspace()

    min_x, min_y, max_x, max_y = get_bounding_box(doc)
    print(f"处理文件 {input_dxf_path} 的边界框：min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")

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

        else:
            # 其他实体类型可根据需要添加
            continue

        entity_dict = {
            'type': entity_type,
            'features': features
        }
        entities.append(entity_dict)

    return entities  # 返回处理后的实体列表

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

    else:
        # 其他未定义的实体类型，保持原样或根据需要处理
        new_features = features

    return {
        'type': entity_type,
        'features': new_features
    }

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

    def to_vector(self, max_len=512):
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

ENTITY_TYPES = ['LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'LEADER', 'INSERT', 'EOS']

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
    'insert_insert_point_x', 'insert_insert_point_y', 'scale_x', 'scale_y', 'insert_rotation'
]


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

    def to_vector(self, max_len=512):
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



def process_dxf_to_vector(input_dxf_path):
    """处理 DXF 文件，返回对应的向量表示"""
    entities = process_dxf_file(input_dxf_path)
    processed_entities = [process_entity(e) for e in entities]
    dxf_seq = DXFSequence.from_dict(processed_entities)
    dxf_seq.normalize()
    vector = dxf_seq.to_vector(max_len=512)  # 保持 max_len=512
    return vector  # 如果超过 max_len，将返回 None

