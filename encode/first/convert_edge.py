# !/user/bin/env python3
# -*- coding: utf-8 -*-
import ezdxf
from calculate_coverage import calculate_layer_bounds

def parse_layer_bounds(layer_bounds):
    # 直接使用calculate_layer_bounds返回的字典
    return layer_bounds

def check_overlap(bounds1, bounds2):
    # 检查两个图层的边界是否重叠
    return not (bounds1['max_x'] < bounds2['min_x'] or bounds1['min_x'] > bounds2['max_x'] or
                bounds1['max_y'] < bounds2['min_y'] or bounds1['min_y'] > bounds2['max_y'])

def build_adjacency_matrix(layer_bounds):
    layers = list(layer_bounds.keys())
    n = len(layers)
    adjacency_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if check_overlap(layer_bounds[layers[i]], layer_bounds[layers[j]]):
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # 无向图

    return layers, adjacency_matrix

def print_adjacency_matrix(layers, adjacency_matrix):
    print("邻接矩阵:")
    print("  ", "  ".join(layers))
    for i, row in enumerate(adjacency_matrix):
        print(layers[i], " ".join(str(x) for x in row))

def convert_to_succs_format(adjacency_matrix):
    # 将邻接矩阵转换为指定的格式
    succs = []
    for row in adjacency_matrix:
        successors = [i for i, val in enumerate(row) if val == 1]
        succs.append(successors)
    return succs

if __name__ == '__main__':
    file_path = r'C:\srtp\datasets\test\one.dxf'

    # 假设calculate_layer_bounds返回图层边界信息
    layer_bounds = calculate_layer_bounds(file_path)

    # 确保layer_bounds的结构是正确的
    # 示例格式:
    # layer_bounds = {
    #     'Layer1': {'min_x': 10, 'max_x': 20, 'min_y': 5, 'max_y': 15},
    #     'Layer2': {'min_x': 15, 'max_x': 25, 'min_y': 10, 'max_y': 20},
    #     ...
    # }

    layers = parse_layer_bounds(layer_bounds)
    layers, adjacency_matrix = build_adjacency_matrix(layer_bounds)
    print("邻接矩阵:")
    print(adjacency_matrix)
    # 转换邻接矩阵为指定格式
    succs = convert_to_succs_format(adjacency_matrix)

    print("邻接矩阵的后继格式:")
    print(succs)
