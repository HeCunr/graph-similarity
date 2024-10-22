# import ezdxf
# import numpy as np
# import json
#
# def dxf_to_json(dxf_file):
#     doc = ezdxf.readfile(dxf_file)
#     msp = doc.modelspace()
#
#     entities = []
#
#     for entity in msp:
#         entity_type = entity.dxftype()
#         entity_params = {}
#
#         if entity_type == "LINE":
#             entity_params["start_point"] = list(entity.dxf.start)
#             entity_params["end_point"] = list(entity.dxf.end)
#         elif entity_type == "CIRCLE":
#             entity_params["center"] = list(entity.dxf.center)
#             entity_params["radius"] = entity.dxf.radius
#         # 处理其他实体类型...
#
#         entities.append({"type": entity_type, "params": entity_params})
#
#     return json.dumps(entities)
#
# def dxf_to_matrix(dxf_file):
#     doc = ezdxf.readfile(dxf_file)
#     msp = doc.modelspace()
#
#     points = []
#
#     for entity in msp:
#         if entity.dxftype() == "LINE":
#             points.append(entity.dxf.start)
#             points.append(entity.dxf.end)
#         elif entity.dxftype() == "CIRCLE":
#             points.append(entity.dxf.center)
#         # 处理其他实体类型...
#
#     points = np.array(points)
#
#     # 计算几何编码矩阵和邻接表
#     # ...
#
#     return geometric_matrix, adjacency_matrix
#
# if __name__ == "__main__":
#     dxf_file = "example.dxf"
#     json_file = "example.json"
#
#     entities_json = dxf_to_json(dxf_file)
#     with open(json_file, "w") as f:
#         f.write(entities_json)
#
#     geometric_matrix, adjacency_matrix = dxf_to_matrix(dxf_file)
#     np.save("geometric_matrix.npy", geometric_matrix)
#     np.save("adjacency_matrix.npy", adjacency_matrix)
#
# # # !/user/bin/env python3
# # # -*- coding: utf-8 -*-
# # # dxf_process.py
# #
# # import ezdxf
# # import numpy as np
# # import json
# #
# # def process_dxf_files(dxf_file):
# #     doc = ezdxf.readfile(dxf_file)
# #     msp = doc.modelspace()
# #
# #     # 提取实体类型和参数信息
# #     entities = []
# #     for e in msp:
# #         entity_type = e.dxftype()
# #         entity_data = e.dxfattribs()
# #         entities.append({"type": entity_type, "data": entity_data})
# #
# #     entities.sort(key=lambda x: x["data"]["handle"])
# #     entity_json = json.dumps(entities)
# #
# #     # 提取几何信息并构建邻接表
# #     vertices = []
# #     edges = []
# #     vertex_map = {}
# #     for e in msp:
# #         if e.dxftype() == "LINE":
# #             start = e.dxf.start
# #             end = e.dxf.end
# #             if str(start) not in vertex_map:
# #                 vertex_map[str(start)] = len(vertices)
# #                 vertices.append(list(start))
# #             if str(end) not in vertex_map:
# #                 vertex_map[str(end)] = len(vertices)
# #                 vertices.append(list(end))
# #             edges.append([vertex_map[str(start)], vertex_map[str(end)]])
# #         elif e.dxftype() == "CIRCLE":
# #             center = e.dxf.center
# #             if str(center) not in vertex_map:
# #                 vertex_map[str(center)] = len(vertices)
# #                 vertices.append(list(center))
# #         # 可以根据需要添加对其他实体类型的支持
# #
# #     geo_encoding = np.array(vertices)
# #     adj_mat = np.zeros((len(vertices), len(vertices)))
# #     for edge in edges:
# #         adj_mat[edge[0], edge[1]] = 1
# #         adj_mat[edge[1], edge[0]] = 1
# #
# #     return {
# #         "geo_encoding": geo_encoding,
# #         "adj_mat": adj_mat,
# #         "entity_json": entity_json
# #     }