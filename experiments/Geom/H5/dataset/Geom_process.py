# -*- coding: utf-8 -*-
"""
dataset/Geom_process.py

功能：
1. 打开给定 DXF 文件，过滤只包含指定的 12 种实体；
2. 若实体数目超过 4096，跳过该文件；
3. 句柄排序并提取特征；
   - 对于每行 (即实体) 的 44 个特征列，若实体 **没有对应的参数**，则保留值为 -1；
   - 若实体有该参数，则进行归一化与量化后赋值；
4. 计算邻接表 succs；
5. 将处理结果以 JSON 字典的形式写入到 "文件名.json" 文件（单行）。
   - 不再包含 "fname" 字段；
   - 新增 "2D-index" 字段，记录每个实体外包盒中心点的归一化量化坐标。
"""
import os
import json
import math
import ezdxf
import numpy as np

# 请根据实际项目的文件结构修改以下导入路径
from dxflib.GeomLib.box import compute_entity_bounding_box
from dataset.Seq_process import (
    get_bounding_box,    # 用于计算整个 DXF 文件的全局边界框
    normalize_coordinate,
    normalize_length,
)

###############################################################################
# 1. 实体类型映射以及特征列下标定义
###############################################################################
ENTITY_TYPE_ORDER = [
    'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT',
    'MTEXT', 'HATCH', 'DIMENSION', 'LEADER', 'INSERT',
    'SPLINE', 'SOLID'
]
ENTITY_TYPE_MAP = {etype: idx for idx, etype in enumerate(ENTITY_TYPE_ORDER)}

# 每一行有 44 个特征列，对应题目给出的注释顺序：
#  0: entity_type
#  1: solid_fill
#  2: associative
#  3: boundary_paths
#  4: text_insert_point_x
#  5: text_insert_point_y
#  6: height
#  7: text_rotation
#  8: mtext_insert_point_x
#  9: mtext_insert_point_y
# 10: char_height
# 11: width
# 12: closed
# 13: points_x
# 14: points_y
# 15: count
# 16: arc_center_x
# 17: arc_center_y
# 18: arc_radius
# 19: start_angle
# 20: end_angle
# 21: start_point_x
# 22: start_point_y
# 23: end_point_x
# 24: end_point_y
# 25: circle_center_x
# 26: circle_center_y
# 27: circle_radius
# 28: defpoint_x
# 29: defpoint_y
# 30: text_midpoint_x
# 31: text_midpoint_y
# 32: vertices_x
# 33: vertices_y
# 34: insert_insert_point_x
# 35: insert_insert_point_y
# 36: scale_x
# 37: scale_y
# 38: insert_rotation
# 39: control_points_x
# 40: control_points_y
# 41: avg_knots
# 42: solid_points_x
# 43: solid_points_y

###############################################################################
# 2. 量化函数：将已经归一化到 [-1, 1] 或经过特殊处理的角度值映射到 [0, 255] 的整数
###############################################################################
def quantize_value(value: float, is_angle: bool = False) -> int:
    """
    将数值从 [-1, 1] (默认) 或 角度(0~360) 范围映射到 [0, 255] 的整数。
    - 若 is_angle=True，则先用 angle/360 => [0,1] 再视为 [-1,1] 后转换。
    - 若超出 [-1,1] 则 clip 到[-1,1] 后再映射。
    """
    if is_angle:
        # 角度范围 [0, 360] -> [0, 1]
        fraction = value / 360.0
        fraction = max(0.0, min(1.0, fraction))
        norm_val = fraction * 2.0 - 1.0  # [0,1] -> [-1,1]
    else:
        # 默认 value 已在 [-1,1]
        norm_val = max(-1.0, min(1.0, value))

    scaled = (norm_val + 1.0) * 0.5 * 255.0
    return int(round(max(0.0, min(255.0, scaled))))

###############################################################################
# 3. 判断两个边界框是否邻接
#    - 如果一个边界框完全包含另一个 => 不邻接
#    - 否则只要相交或贴边 => 邻接
###############################################################################
def box_contained_in(box1, box2) -> bool:
    """
    检查 box1 是否被 box2 严格包含（不含贴边）。
    box 结构： (min_x, min_y, max_x, max_y)
    """
    (min1x, min1y, max1x, max1y) = box1
    (min2x, min2y, max2x, max2y) = box2
    return (min1x > min2x and min1y > min2y and
            max1x < max2x and max1y < max2y)

def boxes_adjacent(box1, box2) -> bool:
    """
    1) 若有完全包含关系 => 不邻接
    2) 否则只要相交或贴边 => 邻接
    """
    if box_contained_in(box1, box2) or box_contained_in(box2, box1):
        return False

    min1x, min1y, max1x, max1y = box1
    min2x, min2y, max2x, max2y = box2
    # 不重叠：x 或 y 方向分离
    if max1x < min2x or min1x > max2x:
        return False
    if max1y < min2y or min1y > max2y:
        return False
    return True

###############################################################################
# 4. 对单个实体进行特征提取：行向量长度为 44
#    若实体无对应参数 => 保持为 -1
###############################################################################
def extract_features(entity, doc,
                     min_x, min_y, max_x, max_y, max_dim) -> list:
    """
    根据题目给出的列顺序，提取并填充所有 44 个特征。
    若实体不具备某参数 => 该位置保持 -1。
    若实体具备该参数 => 执行归一化 + 量化后覆盖。
    """
    # 先全部初始化为 -1
    row = [-1]*44

    etype = entity.dxftype()
    # 如果实体类型不在 12 种列表内，直接返回全 -1
    if etype not in ENTITY_TYPE_MAP:
        return row

    # 第0列放实体类型编码
    row[0] = ENTITY_TYPE_MAP[etype]

    def normX(x):
        """坐标归一化[-1,1]后量化[0,255]"""
        return quantize_value(normalize_coordinate(x, min_x, max_x))

    def normY(y):
        return quantize_value(normalize_coordinate(y, min_y, max_y))

    def normLen(v):
        return quantize_value(normalize_length(v, max_dim))

    def normAngle(a):
        return quantize_value(a, is_angle=True)

    # 对每种实体的特征进行处理：
    try:
        #--------------------- HATCH ---------------------
        if etype == 'HATCH':
            if hasattr(entity.dxf, 'solid_fill'):
                row[1] = int(entity.dxf.solid_fill)
            if hasattr(entity.dxf, 'associative'):
                row[2] = int(entity.dxf.associative)
            if hasattr(entity, 'paths'):
                boundary_paths = len(entity.paths)
                row[3] = min(255, boundary_paths)

        #--------------------- TEXT ----------------------
        elif etype == 'TEXT':
            if hasattr(entity.dxf, 'insert'):
                ins = entity.dxf.insert
                row[4] = normX(ins[0])
                row[5] = normY(ins[1])
            if hasattr(entity.dxf, 'height'):
                row[6] = normLen(entity.dxf.height)
            if hasattr(entity.dxf, 'rotation'):
                row[7] = normAngle(entity.dxf.rotation)

        #--------------------- MTEXT ---------------------
        elif etype == 'MTEXT':
            if hasattr(entity.dxf, 'insert'):
                ins = entity.dxf.insert
                row[8]  = normX(ins[0])
                row[9]  = normY(ins[1])
            if hasattr(entity.dxf, 'char_height'):
                row[10] = normLen(entity.dxf.char_height)
            if hasattr(entity.dxf, 'width'):
                row[11] = normLen(entity.dxf.width)

        #------------------- LWPOLYLINE ------------------
        elif etype == 'LWPOLYLINE':
            if hasattr(entity, 'closed'):
                row[12] = int(entity.closed)
            if hasattr(entity, 'get_points'):
                points = list(entity.get_points())
                count_ = len(points)
                row[15] = min(255, count_)
                if count_ > 0:
                    x_sum = sum(p[0] for p in points)
                    y_sum = sum(p[1] for p in points)
                    avg_x = x_sum / count_
                    avg_y = y_sum / count_
                    row[13] = normX(avg_x)
                    row[14] = normY(avg_y)

        #---------------------- ARC ----------------------
        elif etype == 'ARC':
            if hasattr(entity.dxf, 'center'):
                cx, cy, _ = entity.dxf.center
                row[16] = normX(cx)
                row[17] = normY(cy)
            if hasattr(entity.dxf, 'radius'):
                row[18] = normLen(entity.dxf.radius)
            if hasattr(entity.dxf, 'start_angle'):
                row[19] = normAngle(entity.dxf.start_angle)
            if hasattr(entity.dxf, 'end_angle'):
                row[20] = normAngle(entity.dxf.end_angle)

        #--------------------- LINE ----------------------
        elif etype == 'LINE':
            if hasattr(entity.dxf, 'start'):
                sp = entity.dxf.start
                row[21] = normX(sp[0])
                row[22] = normY(sp[1])
            if hasattr(entity.dxf, 'end'):
                ep = entity.dxf.end
                row[23] = normX(ep[0])
                row[24] = normY(ep[1])

        #-------------------- CIRCLE ---------------------
        elif etype == 'CIRCLE':
            if hasattr(entity.dxf, 'center'):
                cx, cy, _ = entity.dxf.center
                row[25] = normX(cx)
                row[26] = normY(cy)
            if hasattr(entity.dxf, 'radius'):
                row[27] = normLen(entity.dxf.radius)

        #------------------- DIMENSION -------------------
        elif etype == 'DIMENSION':
            if hasattr(entity.dxf, 'defpoint'):
                dx, dy, _ = entity.dxf.defpoint
                row[28] = normX(dx)
                row[29] = normY(dy)
            if hasattr(entity.dxf, 'text_midpoint'):
                mx, my, _ = entity.dxf.text_midpoint
                row[30] = normX(mx)
                row[31] = normY(my)

        #--------------------- LEADER --------------------
        elif etype == 'LEADER':
            if hasattr(entity, 'vertices'):
                vertices = entity.vertices
                if len(vertices) > 0:
                    x_sum = sum(v[0] for v in vertices)
                    y_sum = sum(v[1] for v in vertices)
                    avg_x = x_sum / len(vertices)
                    avg_y = y_sum / len(vertices)
                    row[32] = normX(avg_x)
                    row[33] = normY(avg_y)

        #--------------------- INSERT --------------------
        elif etype == 'INSERT':
            if hasattr(entity.dxf, 'insert'):
                ins = entity.dxf.insert
                row[34] = normX(ins[0])
                row[35] = normY(ins[1])
            if hasattr(entity.dxf, 'xscale'):
                sx = entity.dxf.xscale
                row[36] = quantize_value(sx)  # 简单量化；若需更复杂归一化可自行修改
            if hasattr(entity.dxf, 'yscale'):
                sy = entity.dxf.yscale
                row[37] = quantize_value(sy)
            if hasattr(entity.dxf, 'rotation'):
                row[38] = normAngle(entity.dxf.rotation)

        #--------------------- SPLINE --------------------
        elif etype == 'SPLINE':
            # 控制点
            if hasattr(entity, 'control_points') and entity.control_points:
                cpoints = entity.control_points
                x_sum = sum(p[0] for p in cpoints)
                y_sum = sum(p[1] for p in cpoints)
                avg_x = x_sum / len(cpoints)
                avg_y = y_sum / len(cpoints)
                row[39] = normX(avg_x)
                row[40] = normY(avg_y)
            # knots
            if hasattr(entity, 'knots') and entity.knots:
                knots = entity.knots
                avg_k = sum(knots)/len(knots)
                # 如果需要先归一化到 [-1,1] 再量化，需自己设定规则
                # 这里简单地 clip 到 [-1,1]
                avg_k = max(-1, min(1, avg_k))
                row[41] = quantize_value(avg_k)

        #--------------------- SOLID ---------------------
        elif etype == 'SOLID':
            # vtx0, vtx1, vtx2, vtx3 (一般前三个)
            vtxs = []
            if hasattr(entity.dxf, 'vtx0'):
                vtxs.append((entity.dxf.vtx0.x, entity.dxf.vtx0.y))
            if hasattr(entity.dxf, 'vtx1'):
                vtxs.append((entity.dxf.vtx1.x, entity.dxf.vtx1.y))
            if hasattr(entity.dxf, 'vtx2'):
                vtxs.append((entity.dxf.vtx2.x, entity.dxf.vtx2.y))
            if hasattr(entity.dxf, 'vtx3'):
                vtxs.append((entity.dxf.vtx3.x, entity.dxf.vtx3.y))
            if len(vtxs) > 0:
                x_sum = sum(p[0] for p in vtxs)
                y_sum = sum(p[1] for p in vtxs)
                avg_x = x_sum / len(vtxs)
                avg_y = y_sum / len(vtxs)
                row[42] = normX(avg_x)
                row[43] = normY(avg_y)

    except Exception as ex:
        # 若提取或归一化中途报错，该行维持已有值（大多是 -1）
        # print(f"[DEBUG] extract_features error: {ex}")
        pass

    return row

###############################################################################
# 5. 主函数：对单个 DXF 文件进行处理并写出结果
###############################################################################
def process_single_dxf(dxf_path: str, output_dir: str = None):
    """
    处理单个 DXF 文件：
    1) 读取并过滤实体；
    2) 若实体数超过 4096，跳过不处理；
    3) 句柄排序并提取特征；
       - 对于某字段若实体没有 => 该位置 -1；
    4) 计算邻接表；
    5) 写出到 "xxx.json"（单行 JSON）。
       - 新增 "2D-index" 字段，每个实体的外包盒中心点(x,y)均归一化+量化到[0,255]。
       - 去掉 "fname"。
    """
    if not os.path.isfile(dxf_path):
        print(f"文件不存在：{dxf_path}")
        return

    # 打开 dxf
    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        print(f"无法读取 DXF 文件: {dxf_path}, 错误: {e}")
        return

    msp = doc.modelspace()
    # 收集指定类型实体
    valid_entities = []
    for e in msp:
        if e.dxftype() in ENTITY_TYPE_MAP:
            valid_entities.append(e)

    # 若实体数超过 4096，跳过
    if len(valid_entities) > 4096:
        print(f"文件 {os.path.basename(dxf_path)} 实体数({len(valid_entities)}) > 4096，跳过。")
        return

    # 按照句柄升序排序
    valid_entities.sort(key=lambda ent: int(ent.dxf.handle, 16))

    # 整个文件的边界框，用于归一化(在 dataset/Seq_process.py 里)
    min_x, min_y, max_x, max_y = get_bounding_box(doc)
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height) if max(width, height) > 1e-9 else 1.0

    # 提取特征
    n = len(valid_entities)
    features = np.zeros((n, 44), dtype=np.int32)  # 先用0占位，稍后再覆盖
    for i, ent in enumerate(valid_entities):
        row = extract_features(ent, doc, min_x, min_y, max_x, max_y, max_dim)
        features[i, :] = row

    # 计算实体边界框 => 用于判断邻接 & 2D-index
    bounding_boxes = []
    for ent in valid_entities:
        box = compute_entity_bounding_box(ent, doc)
        if box is None:
            bounding_boxes.append((0, 0, 0, 0))  # 如果无法获取外包盒
        else:
            bounding_boxes.append(box)

    # 构建邻接表
    succs = [[] for _ in range(n)]
    for i in range(n):
        box_i = bounding_boxes[i]
        for j in range(i+1, n):
            box_j = bounding_boxes[j]
            if boxes_adjacent(box_i, box_j):
                succs[i].append(j)
                succs[j].append(i)

    # 计算 "2D-index"：取外包盒中心点 (cx, cy)，归一化并量化到 [0,255]
    def normX(x):
        return quantize_value(normalize_coordinate(x, min_x, max_x))

    def normY(y):
        return quantize_value(normalize_coordinate(y, min_y, max_y))

    two_d_index = []
    for (bx_min, by_min, bx_max, by_max) in bounding_boxes:
        cx = (bx_min + bx_max) / 2.0
        cy = (by_min + by_max) / 2.0
        two_d_index.append([normX(cx), normY(cy)])

    # 组装输出
    dxf_name = os.path.basename(dxf_path)
    result_dict = {
        "src": dxf_name,
        "n_num": n,
        "succs": succs,
        "features": features.tolist(),
        # 去掉 "fname", 增加 "2D-index"
        "2D-index": two_d_index
    }

    if not output_dir:
        output_dir = os.path.dirname(dxf_path)

    base_name = os.path.splitext(dxf_name)[0]
    out_json_path = os.path.join(output_dir, f"{base_name}.json")

    # 写单行 JSON
    with open(out_json_path, "w", encoding="utf-8") as f:
        line = json.dumps(result_dict, ensure_ascii=False)
        f.write(line + "\n")

    print(f"已处理 {dxf_path} => {out_json_path}")

###############################################################################
# 6. 如果你需要批量处理某个目录下的 DXF，可在下方添加示例
###############################################################################
if __name__ == "__main__":
    import glob

    input_dir = r"/home/vllm/encode/data/TRAIN"  # TODO: 修改为实际 DXF 目录
    output_dir = r"/home/vllm/encode/data/Geom/TRAIN_4096"  # TODO: 修改为输出 json 目录

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dxf_files = glob.glob(os.path.join(input_dir, "*.dxf"))
    for dxf_file in dxf_files:
        process_single_dxf(dxf_file, output_dir)
