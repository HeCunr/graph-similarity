# !/user/bin/env python3
# -*- coding: utf-8 -*-
import ezdxf
import os
from ezdxf.entities import BoundaryPathType

def calculate_layer_bounds(file_path):
    doc = ezdxf.readfile(file_path)
    modelspace = doc.modelspace()

    layer_bounds = {}

    for entity in modelspace:
        layer = entity.dxf.layer

        if layer not in layer_bounds:
            layer_bounds[layer] = {
                'min_x': float('inf'),
                'max_x': float('-inf'),
                'min_y': float('inf'),
                'max_y': float('-inf')
            }

        points = []

        if entity.dxftype() == 'LINE':
            points = [entity.dxf.start, entity.dxf.end]

        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            points = [
                (center[0] - radius, center[1]),
                (center[0] + radius, center[1]),
                (center[0], center[1] - radius),
                (center[0], center[1] + radius)
            ]

        elif entity.dxftype() == 'INSERT':
            points = [entity.dxf.insert]

        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points('xy')

        elif entity.dxftype() in ['TEXT', 'MTEXT']:
            points = [entity.dxf.insert]

        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            points = [
                (center[0] + radius, center[1]),
                (center[0] - radius, center[1])
            ]

        elif entity.dxftype() == 'SPLINE':
            points = entity.fit_points  # 使用拟合点属性获取样条曲线的点

        # elif entity.dxftype() == 'POLYLINE':
        #     points = entity.get_points('xy')

        # elif entity.dxftype() == 'ELLIPSE':
        #     center = entity.dxf.center
        #     major_axis = entity.dxf.major_axis
        #     points = [
        #         (center[0] + major_axis[0], center[1] + major_axis[1]),
        #         (center[0] - major_axis[0], center[1] - major_axis[1])
        #     ]

        elif entity.dxftype() == 'DIMENSION':
            points = [entity.dxf.text_midpoint]

        elif entity.dxftype() == 'LEADER':
            points = entity.vertices

        elif entity.dxftype() == 'HATCH':
            for boundary_path in entity.paths:
                if boundary_path.type == BoundaryPathType.POLYLINE:
                    points.extend(boundary_path.vertices)
                elif boundary_path.type == BoundaryPathType.EDGE:
                    for edge in boundary_path.edges:
                        if edge.type == ezdxf.entities.EdgeType.LINE:
                            points.append(edge.start)
                        elif edge.type == ezdxf.entities.EdgeType.ARC:
                            points.append(edge.start_point)

        # 更新该图层的边界
        for point in points:
            layer_bounds[layer]['min_x'] = min(layer_bounds[layer]['min_x'], point[0])
            layer_bounds[layer]['max_x'] = max(layer_bounds[layer]['max_x'], point[0])
            layer_bounds[layer]['min_y'] = min(layer_bounds[layer]['min_y'], point[1])
            layer_bounds[layer]['max_y'] = max(layer_bounds[layer]['max_y'], point[1])

    return layer_bounds

def print_layer_bounds(layer_bounds):
    for layer, bounds in layer_bounds.items():
        print(f"Layer: {layer}")
        print(f"  Min X: {bounds['min_x']}, Max X: {bounds['max_x']}")
        print(f"  Min Y: {bounds['min_y']}, Max Y: {bounds['max_y']}")

if __name__ == '__main__':
    file_path = r'C:\srtp\encode\datasets\4.dxf'
    layer_bounds = calculate_layer_bounds(file_path)
    print_layer_bounds(layer_bounds)




