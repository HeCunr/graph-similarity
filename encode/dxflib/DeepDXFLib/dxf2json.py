import ezdxf
import json
from collections import defaultdict

def to_2d(point):
    """Convert a point to 2D by removing the z-coordinate if present."""
    if isinstance(point, (tuple, list)):
        return tuple(point[:2])
    elif hasattr(point, 'x') and hasattr(point, 'y'):
        return (point.x, point.y)
    elif isinstance(point, (int, float)):
        return (point, 0.0)  # Assuming it's an x-coordinate
    else:
        raise TypeError(f"Unsupported point type: {type(point)}")

def extract_insert_features(entity):
    return {
        'insert_point': to_2d(entity.dxf.insert),
        'scale': (entity.dxf.xscale, entity.dxf.yscale),
        'rotation': entity.dxf.rotation
    }

def extract_line_features(entity):
    return {
        'start_point': to_2d(entity.dxf.start),
        'end_point': to_2d(entity.dxf.end)
    }

def extract_text_features(entity):
    return {
        'insert_point': to_2d(entity.dxf.insert),
        'height': entity.dxf.height,
        'rotation': entity.dxf.rotation
    }

def extract_mtext_features(entity):
    return {
        'insert_point': to_2d(entity.dxf.insert),
        'char_height': entity.dxf.char_height,
        'width': entity.dxf.width,
        'rotation': entity.dxf.rotation
    }

def extract_hatch_features(entity):
    return {
        'solid_fill': entity.dxf.solid_fill,
        'associative': entity.dxf.associative,
        'boundary_paths': len(entity.paths)
    }

def extract_lwpolyline_features(entity):
    return {
        'closed': entity.closed,
        'points': [to_2d(point) for point in entity.get_points()],
        'count': len(entity)
    }

def extract_leader_features(entity):
    return {
        'vertices': [to_2d(vertex) for vertex in entity.vertices]
    }

def extract_circle_features(entity):
    return {
        'center': to_2d(entity.dxf.center),
        'radius': entity.dxf.radius
    }

def extract_dimension_features(entity):
    return {
        'defpoint': to_2d(entity.dxf.defpoint),
        'text_midpoint': to_2d(entity.dxf.text_midpoint)
    }

def extract_arc_features(entity):
    return {
        'center': to_2d(entity.dxf.center),
        'radius': entity.dxf.radius,
        'start_angle': entity.dxf.start_angle,
        'end_angle': entity.dxf.end_angle
    }

def extract_entity_features(entity):
    entity_type = entity.dxftype()
    extractors = {
        'INSERT': extract_insert_features,
        'LINE': extract_line_features,
        'TEXT': extract_text_features,
        'MTEXT': extract_mtext_features,
        'HATCH': extract_hatch_features,
        'LWPOLYLINE': extract_lwpolyline_features,
        'LEADER': extract_leader_features,
        'CIRCLE': extract_circle_features,
        'DIMENSION': extract_dimension_features,
        'ARC': extract_arc_features
    }

    if entity_type in extractors:
        return extractors[entity_type](entity)
    else:
        return {'error': f'Unsupported entity type: {entity_type}'}

def get_normalized_entity_handles(doc):
    msp = doc.modelspace()
    entity_handles = []

    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type in ['ARC', 'TEXT', 'MTEXT', 'LWPOLYLINE', 'INSERT', 'DIMENSION', 'LEADER', 'CIRCLE', 'HATCH', 'LINE']:
            handle = int(entity.dxf.handle, 16)
            entity_handles.append((entity_type, handle, entity))

    # Sort by handle value
    entity_handles.sort(key=lambda x: x[1])

    # Normalize handles
    min_handle = min(h for _, h, _ in entity_handles)
    max_handle = max(h for _, h, _ in entity_handles)
    normalized_handles = [(t, (h - min_handle) / (max_handle - min_handle), e) for t, h, e in entity_handles]

    return normalized_handles

def process_dxf_file(file_path):
    doc = ezdxf.readfile(file_path)
    normalized_entities = get_normalized_entity_handles(doc)

    entities = []
    for entity_type, normalized_handle, entity in normalized_entities:
        try:
            features = extract_entity_features(entity)
            entities.append({
                'type': entity_type,
               # 'normalized_handle': normalized_handle,
                'features': features
            })
        except Exception as e:
            print(f"Error processing entity {entity_type} with handle {entity.dxf.handle}: {str(e)}")

    return entities

def main(dxf_file_path, output_json_path):
    extracted_features = process_dxf_file(dxf_file_path)

    with open(output_json_path, 'w') as f:
        json.dump(extracted_features, f, indent=2)

    print(f"Extracted features have been saved to {output_json_path}")


if __name__ == "__main__":
    dxf_file_path =r'C:\Users\15653\dwg-cx\dataset\modified\DFN6TLCT(NiPdAu)（321） -551 Rev1_3.dxf'  # 请替换为您的DXF文件路径
    output_json_path = r'C:\srtp\encode\data\dxf_json\1_4.json'# 输出 JSON 文件路径
    main(dxf_file_path, output_json_path)



