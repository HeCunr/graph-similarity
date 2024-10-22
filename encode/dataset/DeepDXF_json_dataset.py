
import json
import numpy as np

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
        vector[0] = ENTITY_TYPES.index(self.type)
        for i, feat in enumerate(FEATURE_NAMES):
            if feat in self.features:
                vector[i+1] = self.features[feat]
        return vector

    def quantize(self, value, is_angle=False):
        if is_angle:
            value = value / 360.0  # Normalize angle
        return np.clip(np.round((value + 1.0) / 2 * 255), 0, 255).astype(np.uint8)

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
        self.features['count'] = np.clip(self.features['count'], 0, 255).astype(np.uint8)

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
        self.features['boundary_paths'] = np.clip(self.features['boundary_paths'], 0, 255).astype(np.uint8)

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
        return np.array(vectors)

# 更新 ENTITY_TYPES，添加 'EOS'
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

def load_dxf(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    dxf_seq = DXFSequence.from_dict(json_data)
    dxf_seq.normalize()
    vector = dxf_seq.to_vector()
    return vector