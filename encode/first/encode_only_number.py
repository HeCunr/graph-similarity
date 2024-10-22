import ezdxf
import numpy as np

class DXFLayerEncoder:
    def __init__(self):
        # 定义所需的实体类型
        self.entity_types = [
            'INSERT', 'LINE', 'TEXT', 'MTEXT', 'HATCH', 'LWPOLYLINE',
            'LEADER', 'CIRCLE', 'DIMENSION', 'ARC', 'SPLINE'
        ]
        self.layer_info = {}

    def count_entities(self, file_path):
        # 读取DXF文件
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()

        # 遍历模型空间中的所有实体
        for entity in msp:
            layer_name = entity.dxf.layer
            if layer_name not in self.layer_info:
                self.layer_info[layer_name] = {
                    'entity_count': {etype: 0 for etype in self.entity_types}
                }

            entity_type = entity.dxftype()
            if entity_type in self.layer_info[layer_name]['entity_count']:
                self.layer_info[layer_name]['entity_count'][entity_type] += 1  # 增加该类别的计数

    def get_feature_matrix(self):
        # 创建特征矩阵
        feature_matrix = []
        for layer, info in self.layer_info.items():
            count_vector = np.array([info['entity_count'][etype] for etype in self.entity_types])
            feature_matrix.append(count_vector)

        return np.array(feature_matrix)  # 转换为NumPy数组

# 示例使用
if __name__ == "__main__":
    file_path = r'C:\srtp\datasets\test\one.dxf'
    encoder = DXFLayerEncoder()
    encoder.count_entities(file_path)
    feature_matrix = encoder.get_feature_matrix()

    print("特征矩阵:")
    print(feature_matrix)
