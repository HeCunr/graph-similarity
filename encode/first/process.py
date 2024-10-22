import os
import json
import re
import ezdxf
import numpy as np
from calculate_coverage import calculate_layer_bounds

class DXFLayerEncoder:
    def __init__(self):
        self.entity_types = [
            'INSERT', 'LINE', 'TEXT', 'MTEXT', 'HATCH', 'LWPOLYLINE',
            'LEADER', 'CIRCLE', 'DIMENSION', 'ARC', 'SPLINE'
        ]
        self.layer_info = {}

    def count_entities(self, file_path):
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()

        for entity in msp:
            layer_name = entity.dxf.layer
            if layer_name not in self.layer_info:
                self.layer_info[layer_name] = {
                    'entity_count': {etype: 0 for etype in self.entity_types}
                }

            entity_type = entity.dxftype()
            if entity_type in self.layer_info[layer_name]['entity_count']:
                self.layer_info[layer_name]['entity_count'][entity_type] += 1

    def get_feature_matrix(self):
        feature_matrix = []
        for layer, info in self.layer_info.items():
            count_vector = np.array([info['entity_count'][etype] for etype in self.entity_types])
            feature_matrix.append(count_vector)
        return np.array(feature_matrix)

def parse_layer_bounds(layer_bounds):
    return layer_bounds

def check_overlap(bounds1, bounds2):
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
                adjacency_matrix[j][i] = 1

    return layers, adjacency_matrix

def convert_to_succs_format(adjacency_matrix):
    succs = []
    for row in adjacency_matrix:
        successors = [i for i, val in enumerate(row) if val == 1]
        succs.append(successors)
    return succs

def save_to_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# def remove_digits_from_filename(filename):
#     # 去除文件名中的数字
#     return re.sub(r'\d+', '', os.path.splitext(filename)[0])
# def remove_digits_from_filename(filename):
#     # 返回文件名中最后一个字符（不包括扩展名）
#     return os.path.splitext(filename)[0][-1]

def remove_digits_from_filename(filename):
    # 返回文件名，不包括扩展名
    return os.path.splitext(filename)[0]

def process_dxf_files_in_directory(directory):
    all_results = []

    for filename in os.listdir(directory):
        if filename.endswith('.dxf'):
            dxf_file_path = os.path.join(directory, filename)
            layer_bounds = calculate_layer_bounds(dxf_file_path)

            layers, adjacency_matrix = build_adjacency_matrix(layer_bounds)

            encoder = DXFLayerEncoder()
            encoder.count_entities(dxf_file_path)
            feature_matrix = encoder.get_feature_matrix().tolist()

            succs = convert_to_succs_format(adjacency_matrix)

            json_data = {
                "src": filename,
                "n_num": len(layers),
                "succs": succs,
                "features": feature_matrix,
                "fname": remove_digits_from_filename(filename)
            }

            all_results.append(json_data)

    return all_results

if __name__ == '__main__':
    directory_path = r'C:\Users\15653\dwg-cx\dataset\modified'
    output_json_file = r'C:\Users\15653\dwg-cx\dataset\modified\split_by_own.json'

    results = process_dxf_files_in_directory(directory_path)

    # 保存所有结果为JSON文件
    save_to_json(results, output_json_file)
    print(f"结果已保存至 {output_json_file}")
