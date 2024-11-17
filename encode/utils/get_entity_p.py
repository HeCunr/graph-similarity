import ezdxf
import re
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

def extract_dimension_text_value(text):
    """提取标注文本中的数字值"""
    if text is None or text == "":
        return None
    # 使用正则表达式匹配数字（包括小数点和负号）
    numbers = re.findall(r'-?\d*\.?\d+', text)
    if numbers:
        # 转换为浮点数并返回第一个匹配的数字
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None
def extract_dimension_features(entity):
    # 获取标注文本内容
    dim_text = entity.dxf.text if entity.dxf.text else entity.get_measurement()
    # 提取数字值
    text_value = extract_dimension_text_value(str(dim_text))

    return {
        'defpoint': entity.dxf.defpoint,
        'text_midpoint': entity.dxf.text_midpoint,
        'dim_type': entity.dimtype,
        'text': dim_text,  # 原始文本
        'value': text_value,  # 提取的数字值
    }
def extract_arc_features(entity):
    return {
        'center': entity.dxf.center,
        'radius': entity.dxf.radius,
        'start_angle': entity.dxf.start_angle,
        'end_angle': entity.dxf.end_angle
    }
# 新添加的SOLID实体特征提取函数
def extract_solid_features(entity):
    return {
        'points': [
            entity.dxf.vtx0,  # 第一个点
            entity.dxf.vtx1,  # 第二个点
            entity.dxf.vtx2,  # 第三个点
            getattr(entity.dxf, 'vtx3', entity.dxf.vtx2)  # 第四个点（如果是三角形，则与第三个点相同）
        ],
        'is_3points': not hasattr(entity.dxf, 'vtx3')  # 判断是否为三点SOLID
    }

# 新添加的SPLINE实体特征提取函数
def extract_spline_features(entity):
    return {
        'degree': entity.dxf.degree,
        'control_points': list(entity.control_points),
        'knots': list(entity.knots)
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
        # 新添加的实体类型判断
    elif entity_type == 'SOLID':
        return extract_solid_features(entity)
    elif entity_type == 'SPLINE':
        return extract_spline_features(entity)
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
    file_path =  r'C:\srtp\241101\QFN(0505-0.50)032-0037 20240513_2.dxf'  # 请替换为您的DXF文件路径
    # file_path = r'C:\srtp\FIRST PAPER\encode\data\DXF\DFN6BU(NiPdAu)-437 Rev1_1.dxf'  # 请替换为您的DXF文件路径
    extracted_features = process_dxf_file(file_path)

    for entity_type, features in extracted_features.items():
        # if entity_type == 'SOLID':
        #     print(f"\n{entity_type} 特征:")
        #     for feature in features:
        #         print(feature)
        # if entity_type == 'SPLINE':
        #     print(f"\n{entity_type} 特征:")
        #     for feature in features:
        #         print(feature)
        #     # 特别输出DIMENSION的信息
        if 'DIMENSION' in extracted_features:
            print("\nDIMENSION 特征:")
            for feature in extracted_features['DIMENSION']:
                print(f"文本内容: {feature['text']}")
                print(f"提取的数值: {feature['value']}")
                print("---")