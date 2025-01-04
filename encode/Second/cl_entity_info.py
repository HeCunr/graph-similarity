import ezdxf
import math
import os
import re
import json
import numpy as np
from ezdxf.entities import BoundaryPathType, EdgeType

'''
确定实体类型为以下十种（经复合实体分解后）
实体参数见函数get_entity_info()

note:
1. LINE 为保证编码唯一性，将x坐标小的作为起点
2. MTEXT 目前未考虑文本内容
3. HATCH 直接用边界框坐标作参数
4. LEADER 只考虑了起始点
'''

# 定义实体类型列表
ENTITY_TYPES = ['LINE', 'SPLINE', 'CIRCLE', 'ARC', 'ELLIPSE', 'MTEXT', 'LEADER', 'HATCH', 'DIMENSION', 'SOLID' ]

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

def get_entity_info(entity, doc):
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
        bbox = compute_entity_bounding_box(entity)
        entity_info = [bbox[0], bbox[1], bbox[2], bbox[3]]

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
            entity_info = [vertices[0][0], vertices[0][1],vertices[1][0], vertices[1][1],vertices[2][0], vertices[2][1]]

    return entity_info