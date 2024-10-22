import ezdxf

def extract_insert_features(entity):
    return {
        'name': entity.dxf.name,
        'insert_point': entity.dxf.insert,
        'scale': (entity.dxf.xscale, entity.dxf.yscale, entity.dxf.zscale),
        'rotation': entity.dxf.rotation
    }

def extract_line_features(entity):
    return {
        'start_point': entity.dxf.start,
        'end_point': entity.dxf.end
    }

def extract_text_features(entity):
    return {
        'text': entity.dxf.text,
        'insert_point': entity.dxf.insert,
        'height': entity.dxf.height,
        'rotation': entity.dxf.rotation
    }

def extract_mtext_features(entity):
    return {
        'text': entity.text,
        'insert_point': entity.dxf.insert,
        'char_height': entity.dxf.char_height,
        'width': entity.dxf.width,
        'rotation': entity.dxf.rotation  # 新添加的旋转属性
    }

def extract_hatch_features(entity):
    return {
        'pattern_name': entity.dxf.pattern_name,
        'solid_fill': entity.dxf.solid_fill,
        'associative': entity.dxf.associative,
        'boundary_paths': len(entity.paths)  # 新添加的边界路径数量
    }

def extract_lwpolyline_features(entity):
    return {
        'closed': entity.closed,
        'points': list(entity.get_points()),
        'count': len(entity)
    }

def extract_leader_features(entity):
    return {
        'vertices': list(entity.vertices),
        'annotation_type': entity.dxf.annotation_type
    }

def extract_circle_features(entity):
    return {
        'center': entity.dxf.center,
        'radius': entity.dxf.radius
    }

def extract_dimension_features(entity):
    return {
        'defpoint': entity.dxf.defpoint,
        'text_midpoint': entity.dxf.text_midpoint,
        'dim_type': entity.dimtype
    }

def extract_arc_features(entity):
    return {
        'center': entity.dxf.center,
        'radius': entity.dxf.radius,
        'start_angle': entity.dxf.start_angle,
        'end_angle': entity.dxf.end_angle
    }

def extract_entity_features(entity):
    entity_type = entity.dxftype()
    if entity_type == 'INSERT':
        return extract_insert_features(entity)
    elif entity_type == 'LINE':
        return extract_line_features(entity)
    elif entity_type == 'TEXT':
        return extract_text_features(entity)
    elif entity_type == 'MTEXT':
        return extract_mtext_features(entity)
    elif entity_type == 'HATCH':
        return extract_hatch_features(entity)
    elif entity_type == 'LWPOLYLINE':
        return extract_lwpolyline_features(entity)
    elif entity_type == 'LEADER':
        return extract_leader_features(entity)
    elif entity_type == 'CIRCLE':
        return extract_circle_features(entity)
    elif entity_type == 'DIMENSION':
        return extract_dimension_features(entity)
    elif entity_type == 'ARC':
        return extract_arc_features(entity)
    else:
        return {'error': f'Unsupported entity type: {entity_type}'}

def process_dxf_file(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    entities = {}
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(extract_entity_features(entity))

    return entities

# 使用示例
if __name__ == "__main__":
    file_path = r'C:\Users\15653\dwg-cx\dataset\modified\DFN6TLCT(NiPdAu)（321） -551 Rev1_2.dxf'  # 请替换为您的DXF文件路径
    extracted_features = process_dxf_file(file_path)

    for entity_type, features in extracted_features.items():
        print(f"\n{entity_type} 特征:")
        for feature in features:
            print(feature)