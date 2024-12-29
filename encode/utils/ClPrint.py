import ezdxf
import math
import os
import re
import json
import numpy as np
from ezdxf.entities import BoundaryPathType, EdgeType

# Keep existing ENTITY_TYPES and helper functions...
ENTITY_TYPES = ['LINE', 'SPLINE', 'CIRCLE', 'ARC', 'ELLIPSE', 'MTEXT', 'LEADER', 'HATCH', 'DIMENSION', 'SOLID']

def parse_text(text):
    match1 = re.match(r'^(\d+\.\d+)$', text)
    if match1:
        text_0 = match1.group(1)
        text_1 = "0"
        return float(text_0), int(text_1)

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

def get_entity_info(entity, doc):
    entity_type = entity.dxftype()
    entity_info = []

    if entity_type == 'LINE':
        start_point = entity.dxf.start
        end_point = entity.dxf.end
        if start_point[0] > end_point[0]:
            start_point, end_point = end_point, start_point
        entity_info = [start_point[0], start_point[1], end_point[0], end_point[1]]

    elif entity_type == 'SPLINE':
        vertices = entity.fit_points
        if len(vertices) >= 2:
            entity_info = [vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1], vertices[-1][0], vertices[-1][1]]

    elif entity_type == 'CIRCLE':
        center = entity.dxf.center
        radius = entity.dxf.radius
        entity_info = [center[0], center[1], radius]

    elif entity_type == 'ARC':
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        entity_info = [center[0], center[1], radius, start_angle, end_angle]

    elif entity_type == 'ELLIPSE':
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio
        entity_info = [center[0], center[1], major_axis[0], major_axis[1], ratio]

    elif entity_type == 'MTEXT':
        insert = entity.dxf.insert
        height = entity.dxf.char_height
        entity_info = [insert[0], insert[1], height]

    elif entity_type == 'HATCH':
        # bbox = compute_entity_bounding_box(entity)
        # entity_info = [bbox[0], bbox[1], bbox[2], bbox[3]]
        pass

    elif entity_type == 'DIMENSION':
        def_point = entity.dxf.defpoint
        midpoint = entity.dxf.text_midpoint
        text = entity.dxf.text

        value_0, value_1 = parse_text(text)
        if value_0 == 0:
            value_0 = extract_line_dim(entity, doc)
        entity_info = [def_point[0], def_point[1], midpoint[0], midpoint[1], value_0, value_1]

    elif entity_type == 'LEADER':
        vertices = entity.vertices
        entity_info = [vertices[0][0], vertices[0][1], vertices[-1][0], vertices[-1][1]]

    elif entity_type == 'SOLID':
        vertices = entity.vertices()
        if len(vertices) >= 2:
            entity_info = [vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1]]

    return entity_type, entity_info

def print_entity_parameters(entity_type, params):
    """Print entity parameters with descriptive labels"""
    print(f"\nEntity Type: {entity_type}")

    if entity_type == 'LINE':
        print(f"Start point: ({params[0]}, {params[1]})")
        print(f"End point: ({params[2]}, {params[3]})")

    elif entity_type == 'SPLINE':
        print(f"Start point: ({params[0]}, {params[1]})")
        print(f"Second point: ({params[2]}, {params[3]})")
        print(f"End point: ({params[4]}, {params[5]})")

    elif entity_type == 'CIRCLE':
        print(f"Center: ({params[0]}, {params[1]})")
        print(f"Radius: {params[2]}")

    elif entity_type == 'ARC':
        print(f"Center: ({params[0]}, {params[1]})")
        print(f"Radius: {params[2]}")
        print(f"Start angle (rad): {params[3]}")
        print(f"End angle (rad): {params[4]}")

    elif entity_type == 'ELLIPSE':
        print(f"Center: ({params[0]}, {params[1]})")
        print(f"Major axis: ({params[2]}, {params[3]})")
        print(f"Ratio: {params[4]}")

    elif entity_type == 'MTEXT':
        print(f"Insertion point: ({params[0]}, {params[1]})")
        print(f"Height: {params[2]}")

    elif entity_type == 'HATCH':
        print(f"Bounding box: ({params[0]}, {params[1]}) to ({params[2]}, {params[3]})")

    elif entity_type == 'DIMENSION':
        #print(f"Definition point: ({params[0]}, {params[1]})")
        #print(f"Text midpoint: ({params[2]}, {params[3]})")
        print(f"Value: {params[4]} ± {params[5]}")

    elif entity_type == 'LEADER':
        print(f"Start point: ({params[0]}, {params[1]})")
        print(f"End point: ({params[2]}, {params[3]})")

    elif entity_type == 'SOLID':
        print(f"Point 1: ({params[0]}, {params[1]})")
        print(f"Point 2: ({params[2]}, {params[3]})")
        print(f"Point 3: ({params[4]}, {params[5]})")

def process_dxf_file(filename):
    """Process a DXF file and print information about all entities"""
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()

    print(f"Processing DXF file: {filename}")
    print("=" * 50)

    for entity in msp:
        entity_type, params = get_entity_info(entity, doc)
        if entity_type == 'DIMENSION':
            print(entity.dxf.text)
            print(params)
        # if entity_type in ENTITY_TYPES and params:
        #     print_entity_parameters(entity_type, params)
        #     print("-" * 30)

if __name__ == '__main__':
    # Example usage
    dxf_file = r'C:\srtp\241101\QFN(0505-0.50)032-0037 20240513_2.dxf'  # Replace with your DXF file path
    process_dxf_file(dxf_file)