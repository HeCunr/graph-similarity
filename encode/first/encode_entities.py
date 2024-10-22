#整个dxf一起编码
# import ezdxf
# import numpy as np
#
# class DXFEntityCounter:
#     def __init__(self):
#         # 定义所需的实体类型
#         self.entity_types = [
#             'INSERT',
#             'LINE',
#             'TEXT',
#             'MTEXT',
#             'HATCH',
#             'LWPOLYLINE',
#             'LEADER',
#             'CIRCLE',
#             'DIMENSION',
#             'ARC',
#             'SPLINE',
#             'ELLIPSE',
#             'POLYLINE'
#         ]
#         # 初始化计数字典
#         self.entity_count = {entity_type: 0 for entity_type in self.entity_types}
#
#     def count_entities(self, file_path):
#         # 读取DXF文件
#         doc = ezdxf.readfile(file_path)
#         msp = doc.modelspace()
#
#         # 遍历模型空间中的所有实体
#         for entity in msp:
#             entity_type = entity.dxftype()
#             if entity_type in self.entity_count:
#                 self.entity_count[entity_type] += 1  # 增加该类别的计数
#
#     def get_normalized_vector(self):
#         # 创建固定维度的向量
#         count_vector = np.array([self.entity_count[entity_type] for entity_type in self.entity_types])
#
#         # 规范化向量
#         max_count = np.max(count_vector) if np.max(count_vector) > 0 else 1  # 避免除以0
#         normalized_vector = count_vector / max_count
#
#         # 扩展维度以适配神经网络输入
#         input_vector = normalized_vector.reshape(1, -1)  # 添加批次维度
#         return input_vector
#
# # 示例使用
# if __name__ == "__main__":
#     file_path = r'C:\srtp\encode\datasets\2.dxf'
#     counter = DXFEntityCounter()
#     counter.count_entities(file_path)
#     input_vector = counter.get_normalized_vector()
#     print("神经网络输入向量:", input_vector)
#----------------------------------------------------------
#考虑颜色，线型，图层
# import ezdxf
# import numpy as np
#
# class DXFLayerEncoder:
#     def __init__(self):
#         # 定义所需的实体类型
#         self.entity_types = [
#             'INSERT', 'LINE', 'TEXT', 'MTEXT', 'HATCH', 'LWPOLYLINE',
#             'LEADER', 'CIRCLE', 'DIMENSION', 'ARC', 'SPLINE', 'ELLIPSE', 'POLYLINE'
#         ]
#         self.layer_info = {}
#
#     def count_entities(self, file_path):
#         # 读取DXF文件
#         doc = ezdxf.readfile(file_path)
#         msp = doc.modelspace()
#
#         # 遍历模型空间中的所有实体
#         for entity in msp:
#             layer_name = entity.dxf.layer
#             if layer_name not in self.layer_info:
#                 self.layer_info[layer_name] = {
#                     'entity_count': {etype: 0 for etype in self.entity_types},
#                     'color': entity.dxf.color,  # 获取图层颜色
#                     'linetype': entity.dxf.linetype  # 获取图层线型
#                 }
#
#             entity_type = entity.dxftype()
#             if entity_type in self.layer_info[layer_name]['entity_count']:
#                 self.layer_info[layer_name]['entity_count'][entity_type] += 1  # 增加该类别的计数
#
#     def get_normalized_vectors(self):
#         # 创建所有图层的编码向量
#         encoded_vectors = {}
#         for layer, info in self.layer_info.items():
#             count_vector = np.array([info['entity_count'][etype] for etype in self.entity_types])
#             max_count = np.max(count_vector) if np.max(count_vector) > 0 else 1  # 避免除以0
#             normalized_vector = count_vector / max_count
#
#             # 添加颜色和线型编码（假设色彩和线型可以进行简单编码）
#             color_encoding = info['color'] / 256  # 假设颜色在0-256范围内
#             linetype_encoding = 1 if info['linetype'] else 0  # 简化处理，若有线型则编码为1
#
#             # 组合编码向量
#             final_vector = np.concatenate((normalized_vector, [color_encoding, linetype_encoding]))
#             encoded_vectors[layer] = final_vector
#
#         return encoded_vectors
#
# # 示例使用
# if __name__ == "__main__":
#     file_path = r'C:\srtp\encode\datasets\2.dxf'
#     encoder = DXFLayerEncoder()
#     encoder.count_entities(file_path)
#     encoded_vectors = encoder.get_normalized_vectors()
#
#     for layer, vector in encoded_vectors.items():
#         # print(f"图层 '{layer}' 编码后的向量:", vector)
#         print(vector)
#----------------------------------------------------------
#只考虑图层
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

    def get_normalized_vectors(self):
        # 创建所有图层的编码向量
        encoded_vectors = {}
        for layer, info in self.layer_info.items():
            count_vector = np.array([info['entity_count'][etype] for etype in self.entity_types])
            max_count = np.max(count_vector) if np.max(count_vector) > 0 else 1  # 避免除以0
            normalized_vector = count_vector / max_count

            # 直接使用归一化向量
            encoded_vectors[layer] = normalized_vector

        return encoded_vectors

# 示例使用
if __name__ == "__main__":
    file_path = r'C:\srtp\datasets\test\one.dxf'
    encoder = DXFLayerEncoder()
    encoder.count_entities(file_path)
    encoded_vectors = encoder.get_normalized_vectors()

    for layer, vector in encoded_vectors.items():
        #print(f"图层 '{layer}' 编码后的向量:", vector)
        print(vector)
