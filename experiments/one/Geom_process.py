# dataset/Geom_process.py

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
# 1. 实体类型映射（1-based）以及列常量定义
###############################################################################
ENTITY_TYPE_MAP_1B = {
    'LINE': 1, 'CIRCLE': 2, 'ARC': 3, 'LWPOLYLINE': 4, 'TEXT': 5,
    'MTEXT': 6, 'HATCH': 7, 'DIMENSION': 8, 'LEADER': 9, 'INSERT': 10,
    'SPLINE': 11, 'SOLID': 12
}

# 我们只关注题目要求的 12 种实体
VALID_ENTITY_TYPES = set(ENTITY_TYPE_MAP_1B.keys())

# 新的 features 每行 14 列的含义
NUM_FEATURES = 14
(
    FEAT_ETYPE,          # 0: entity_type (1~12)
    FEAT_PATCH_ID,       # 1: Patch_Id (0~255)
    FEAT_FLAGS,          # 2: flags
    FEAT_BBOX_CX,        # 3: bbox_center_x
    FEAT_BBOX_CY,        # 4: bbox_center_y
    FEAT_BBOX_W,         # 5: bbox_width
    FEAT_BBOX_H,         # 6: bbox_height
    FEAT_MAIN_ANGLE,     # 7: main_angle
    FEAT_SECOND_ANGLE,   # 8: second_angle
    FEAT_MAIN_LEN,       # 9: main_length
    FEAT_ENTITY_X,       # 10: entity_x
    FEAT_ENTITY_Y,       # 11: entity_y
    FEAT_PARAMA,         # 12: ParamA
    FEAT_PARAMB          # 13: ParamB
) = range(NUM_FEATURES)


###############################################################################
# 2. 量化辅助函数
###############################################################################
def quantize_value(value: float, is_angle: bool = False) -> int:
    """
    将数值映射到 [0, 255] 的整数。
    - 若 is_angle=True，angle 允许范围是 [-360, 360]；先除以 360 => [-1,1]，再映射到 [0,255]。
    - 若 is_angle=False，假定 value 已在 [-1,1] 或 0~正值范围，先clip到[-1,1]再映射。
      （这里和原逻辑保持一致，也可按需求自行修改）
    """
    if is_angle:
        # 角度范围 [-360, 360] => fraction in [-1,1]
        fraction = value / 360.0
        # clip
        fraction = max(-1.0, min(1.0, fraction))
        norm_val = fraction  # [-1,1]
    else:
        # 默认 clip 到[-1,1]
        norm_val = max(-1.0, min(1.0, value))

    scaled = (norm_val + 1.0) * 0.5 * 255.0
    return int(round(max(0.0, min(255.0, scaled))))


def quantize_nonneg_int(value: int) -> int:
    """
    对非负整数直接 clip 到 [0, 255]。
    例如 boundary_paths, count 等。
    """
    return max(0, min(255, value))


###############################################################################
# 3. 邻接判定函数 (与原先相同)
###############################################################################
def box_contained_in(box1, box2) -> bool:
    (min1x, min1y, max1x, max1y) = box1
    (min2x, min2y, max2x, max2y) = box2
    return (min1x > min2x and min1y > min2y and
            max1x < max2x and max1y < max2y)

def boxes_adjacent(box1, box2) -> bool:
    if box_contained_in(box1, box2) or box_contained_in(box2, box1):
        return False
    min1x, min1y, max1x, max1y = box1
    min2x, min2y, max2x, max2y = box2
    if max1x < min2x or min1x > max2x:
        return False
    if max1y < min2y or min1y > max2y:
        return False
    return True


###############################################################################
# 4. 计算 Patch_Id: 将全局 bbox 分成 16×16，共 256 个网格
###############################################################################
def compute_patch_id(cx, cy, min_x, min_y, max_x, max_y):
    """
    按从左到右、从上到下，将 bounding box 分割成16×16=256个格子。
    row=0 表示最上侧，row=15 表示最下侧；col=0表示最左，col=15表示最右。
    patch_id = row*16 + col
    """
    width = max_x - min_x
    height = max_y - min_y
    if width < 1e-9 or height < 1e-9:
        # 如果整个图过小，就直接放到0号格子。
        return 0

    # col 从左到右
    col = int(16 * (cx - min_x) / width)
    # row 从上到下 => (max_y - cy)
    row = int(16 * (max_y - cy) / height)

    # clip 到 [0, 15]
    col = max(0, min(15, col))
    row = max(0, min(15, row))

    patch_id = row * 16 + col
    return patch_id


###############################################################################
# 5. 计算各种字段: flags、main_angle、second_angle、main_length 等
###############################################################################
def compute_flags(etype, entity, is_closed, solid_fill, associative, has_neighbors):
    """
    根据题意计算 bit0~bit7 的组合。
    """
    # bit0 => 是否闭合 (仅LWPOLYLINE)
    bit0 = 1 if (etype == 'LWPOLYLINE' and is_closed) else 0
    # bit1 => 是否实体填充 (仅HATCH, solid_fill=1)
    bit1 = 1 if (etype == 'HATCH' and solid_fill == 1) else 0
    # bit2 => 是否关联填充 (仅HATCH, associative=1)
    bit2 = 1 if (etype == 'HATCH' and associative == 1) else 0
    # bit3 => 是否文本类 (TEXT, MTEXT, DIMENSION)
    bit3 = 1 if etype in ('TEXT','MTEXT','DIMENSION') else 0
    # bit4 => 是否孤立实体 (没有邻接)
    bit4 = 0 if has_neighbors else 1
    # bit5 => 是否填充 main_angle & second_angle
    bit5 = 1 if etype in ('LINE','ARC','CIRCLE','TEXT','MTEXT','INSERT','LWPOLYLINE') else 0
    # bit6 => 是否需要计算平均点 (LWPOLYLINE, LEADER, SPLINE, SOLID)
    bit6 = 1 if etype in ('LWPOLYLINE','LEADER','SPLINE','SOLID') else 0
    # bit7 => 是否需要 ParamA/ParamB (INSERT, DIMENSION, MTEXT)
    bit7 = 1 if etype in ('INSERT','DIMENSION','MTEXT') else 0

    flags = (bit0 << 0) | (bit1 << 1) | (bit2 << 2) | (bit3 << 3) | \
            (bit4 << 4) | (bit5 << 5) | (bit6 << 6) | (bit7 << 7)
    return flags


def compute_main_angle(etype, entity):
    """
    main_angle 的定义：
     - LINE => start->end 的方向角度(°)，范围[-180,180]也可视为[-360,360]
     - ARC => start_angle
     - CIRCLE => 0
     - TEXT => text_rotation
     - MTEXT => 0
     - INSERT => insert_rotation
     - LWPOLYLINE => 0
     - 其他 => 0
    """
    try:
        if etype == 'LINE':
            sp = entity.dxf.start
            ep = entity.dxf.end
            dx = ep[0] - sp[0]
            dy = ep[1] - sp[1]
            angle_deg = math.degrees(math.atan2(dy, dx))  # -180~180
            return angle_deg
        elif etype == 'ARC':
            return entity.dxf.start_angle
        elif etype == 'CIRCLE':
            return 0
        elif etype == 'TEXT':
            return entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0
        elif etype == 'INSERT':
            return entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0
        elif etype == 'LWPOLYLINE':
            return 0
        else:
            # MTEXT => 0, DIMENSION => 0, HATCH => 0, ...
            return 0
    except:
        return 0


def compute_second_angle(etype, entity, main_angle, is_closed):
    """
    second_angle 的定义：
     - LINE => 与 main_angle 相同
     - ARC => end_angle
     - CIRCLE => 360
     - LWPOLYLINE => 若 closed=1 => 255(量化后)，先给360做量化即可; 否则0
     - 其他 => 0
    """
    try:
        if etype == 'LINE':
            return main_angle  # same as main_angle
        elif etype == 'ARC':
            return entity.dxf.end_angle
        elif etype == 'CIRCLE':
            return 360
        elif etype == 'LWPOLYLINE':
            return 360 if is_closed else 0
        else:
            return 0
    except:
        return 0


def compute_main_length(etype, entity, doc, max_dim):
    """
    main_length 的定义 (归一化后量化到[0,255]):
     - LINE => 线段长度
     - CIRCLE => 半径
     - ARC => 半径
     - LWPOLYLINE => count (顶点数，直接clip到255)
     - TEXT => height
     - MTEXT => width
     - SPLINE => avg_knots
     - HATCH => boundary_paths
     - 其他 => 0
    """
    try:
        if etype == 'LINE':
            sp = entity.dxf.start
            ep = entity.dxf.end
            length = math.hypot(ep[0]-sp[0], ep[1]-sp[1])
            return quantize_value(normalize_length(length, max_dim))
        elif etype == 'CIRCLE':
            r = entity.dxf.radius
            return quantize_value(normalize_length(r, max_dim))
        elif etype == 'ARC':
            r = entity.dxf.radius
            return quantize_value(normalize_length(r, max_dim))
        elif etype == 'LWPOLYLINE':
            points = list(entity.get_points())
            count_ = len(points)
            return quantize_nonneg_int(count_)
        elif etype == 'TEXT':
            h = entity.dxf.height if hasattr(entity.dxf, 'height') else 0
            return quantize_value(normalize_length(h, max_dim))
        elif etype == 'MTEXT':
            w = entity.dxf.width if hasattr(entity.dxf, 'width') else 0
            return quantize_value(normalize_length(w, max_dim))
        elif etype == 'SPLINE':
            # avg_knots
            if hasattr(entity, 'knots') and entity.knots:
                ks = entity.knots
                avg_k = sum(ks)/len(ks)
                # clip到[-1,1]再量化
                avg_k = max(-1.0, min(1.0, avg_k))
                return quantize_value(avg_k)
            else:
                return 0
        elif etype == 'HATCH':
            # boundary_paths
            num_paths = len(entity.paths) if hasattr(entity, 'paths') else 0
            return quantize_nonneg_int(num_paths)
        else:
            return 0
    except:
        return 0


def get_entity_xy(etype, entity):
    """
    entity_x, entity_y 的取值:
     TEXT => text_insert_point_x, text_insert_point_y
     MTEXT => mtext_insert_point_x, mtext_insert_point_y
     LWPOLYLINE => points_x, points_y (平均顶点)
     ARC => arc_center_x, arc_center_y
     CIRCLE => circle_center_x, circle_center_y
     LEADER => vertices_x, vertices_y (平均)
     INSERT => insert_insert_point_x, insert_insert_point_y
     SPLINE => control_points_x, control_points_y (平均)
     DIMENSION => defpoint_x, defpoint_y
     SOLID => solid_points_x, solid_points_y (平均)
     其他 => 0,0
    这里假设在 extract_features 或类似地方已经计算好了
    不过现在我们按原逻辑:
    - TEXT: entity.dxf.insert
    - ARC: entity.dxf.center
    ...
    但我们只在主函数统一提取一次即可(见下文), 再传进来.
    """
    return 0.0, 0.0  # 在后面统一处理，这里留空


def compute_paramA(etype, entity, doc, max_dim):
    """
    ParamA:
     - DIMENSION => text_midpoint_x
     - INSERT => scale_x (简单clip到[-1,1]后量化?)
     - MTEXT => char_height
     - 其他 => 0
    """
    try:
        if etype == 'DIMENSION':
            if hasattr(entity.dxf, 'text_midpoint'):
                return entity.dxf.text_midpoint[0]
            else:
                return 0
        elif etype == 'INSERT':
            sx = entity.dxf.xscale if hasattr(entity.dxf, 'xscale') else 1.0
            # 如果你需要更精确的归一化，可自行处理；这里先当做 [-1,1] 之外的也clip
            # 通常缩放因子>=0，但这里也不严格区分
            # 也可仅做 clip [0,255] => quantize_nonneg_int(int(sx)) 视需求而定
            # 为了和原 angle/length逻辑一致，这里做“normalize_coordinate”并量化
            # 也可直接 min(255, round(sx)) =>看具体需求
            # 这里示例：简单地把 scale 视为 [-1,1] => [0,255]。若>1则clip=1
            sx_ = max(-1.0, min(1.0, sx))
            return sx_
        elif etype == 'MTEXT':
            # char_height
            ch = entity.dxf.char_height if hasattr(entity.dxf, 'char_height') else 0
            # 先当做 length
            return normalize_length(ch, max_dim)
        else:
            return 0
    except:
        return 0


def compute_paramB(etype, entity, doc, max_dim):
    """
    ParamB:
     - DIMENSION => text_midpoint_y
     - INSERT => scale_y
     - 其他 => 0
    """
    try:
        if etype == 'DIMENSION':
            if hasattr(entity.dxf, 'text_midpoint'):
                return entity.dxf.text_midpoint[1]
            else:
                return 0
        elif etype == 'INSERT':
            sy = entity.dxf.yscale if hasattr(entity.dxf, 'yscale') else 1.0
            sy_ = max(-1.0, min(1.0, sy))
            return sy_
        else:
            return 0
    except:
        return 0


###############################################################################
# 6. 主函数：对单个 DXF 文件进行处理并写出结果
###############################################################################
def process_single_dxf(dxf_path: str, output_dir: str = None):
    """
    改动重点：
      1) 不再按句柄排序，而是先计算bbox->邻接->再根据 bbox中心排序。
      2) 最终 "features" 维度是 n×14。
      3) 保留 succs，但需要根据新排序重排。
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
        if e.dxftype() in VALID_ENTITY_TYPES:
            valid_entities.append(e)

    # 若实体数超过 4096，跳过
    if len(valid_entities) > 4096:
        print(f"文件 {os.path.basename(dxf_path)} 实体数({len(valid_entities)}) > 4096，跳过。")
        return

    n = len(valid_entities)
    # 整个文件的全局 bounding box
    min_x, min_y, max_x, max_y = get_bounding_box(doc)
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height) if max(width, height) > 1e-9 else 1.0

    # 1) 计算每个实体的bbox、中心点，以及后面要用的各种信息
    bounding_boxes = []
    centers = []
    entity_infos = []
    for i, ent in enumerate(valid_entities):
        box = compute_entity_bounding_box(ent, doc)
        if box is None:
            box = (0,0,0,0)
        bxmin, bymin, bxmax, bymax = box
        cx = 0.5*(bxmin+bxmax)
        cy = 0.5*(bymin+bymax)
        bounding_boxes.append(box)
        centers.append((cx, cy))

        entity_infos.append({
            'entity': ent,
            'etype': ent.dxftype(),
            'box': box,
            'center': (cx, cy),
        })

    # 2) 构建邻接表 (先按原顺序 i=0..n-1)
    succs = [[] for _ in range(n)]
    for i in range(n):
        box_i = bounding_boxes[i]
        for j in range(i+1, n):
            box_j = bounding_boxes[j]
            if boxes_adjacent(box_i, box_j):
                succs[i].append(j)
                succs[j].append(i)

    # 3) 根据 (cx升序, cy降序) 排序 => 得到新顺序
    order = sorted(range(n), key=lambda i: (centers[i][0], -centers[i][1]))
    old2new = {}
    for new_i, old_i in enumerate(order):
        old2new[old_i] = new_i

    # 4) 根据新顺序重排 succs
    new_succs = [[] for _ in range(n)]
    for old_i in range(n):
        new_i = old2new[old_i]
        for old_j in succs[old_i]:
            new_j = old2new[old_j]
            new_succs[new_i].append(new_j)

    # 5) 生成特征矩阵 n×14
    features = np.zeros((n, NUM_FEATURES), dtype=np.int32)

    for new_i, old_i in enumerate(order):
        ent_info = entity_infos[old_i]
        ent = ent_info['entity']
        etype = ent_info['etype']
        box = ent_info['box']
        cx, cy = ent_info['center']
        bxmin, bymin, bxmax, bymax = box

        # 取 1-based entity_type
        ecode = ENTITY_TYPE_MAP_1B.get(etype, 0)

        # Patch_Id
        patch_id = compute_patch_id(cx, cy, min_x, min_y, max_x, max_y)
        patch_id = max(0, min(255, patch_id))  # clip

        # 是否闭合
        is_closed = False
        if etype == 'LWPOLYLINE' and hasattr(ent, 'closed'):
            is_closed = bool(ent.closed)

        # solid_fill, associative
        sf = 0
        assoc = 0
        if etype == 'HATCH':
            if hasattr(ent.dxf, 'solid_fill'):
                sf = int(ent.dxf.solid_fill)
            if hasattr(ent.dxf, 'associative'):
                assoc = int(ent.dxf.associative)

        # 是否有邻接
        neighbor_count = len(new_succs[new_i])
        has_neighbors = (neighbor_count > 0)

        # flags
        flags_val = compute_flags(etype, ent, is_closed, sf, assoc, has_neighbors)

        # bbox中心量化 [0,255]
        q_cx = quantize_value(normalize_coordinate(cx, min_x, max_x))
        q_cy = quantize_value(normalize_coordinate(cy, min_y, max_y))

        # bbox宽高量化
        bw = bxmax - bxmin
        bh = bymax - bymin
        q_bw = quantize_value(normalize_length(bw, max_dim))
        q_bh = quantize_value(normalize_length(bh, max_dim))

        # main_angle, second_angle
        angle1 = compute_main_angle(etype, ent)
        angle2 = compute_second_angle(etype, ent, angle1, is_closed)

        q_angle1 = quantize_value(angle1, is_angle=True)
        q_angle2 = quantize_value(angle2, is_angle=True)

        # main_length
        q_main_len = compute_main_length(etype, ent, doc, max_dim)

        # entity_x, entity_y
        # 在原extract_features中:
        #   TEXT => text_insert_point_x => normX;   ...
        # 这里直接取其平均点/插入点等
        ex = 0
        ey = 0
        try:
            if etype == 'TEXT':
                ins = ent.dxf.insert
                ex = ins[0]
                ey = ins[1]
            elif etype == 'MTEXT':
                ins = ent.dxf.insert
                ex = ins[0]
                ey = ins[1]
            elif etype == 'LWPOLYLINE':
                # 取平均x,y
                pts = list(ent.get_points())
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    ex = sum(xs)/len(xs)
                    ey = sum(ys)/len(ys)
            elif etype == 'ARC':
                cx_, cy_, _ = ent.dxf.center
                ex = cx_
                ey = cy_
            elif etype == 'CIRCLE':
                cx_, cy_, _ = ent.dxf.center
                ex = cx_
                ey = cy_
            elif etype == 'LEADER':
                vs = ent.vertices
                if vs:
                    xs = [v[0] for v in vs]
                    ys = [v[1] for v in vs]
                    ex = sum(xs)/len(xs)
                    ey = sum(ys)/len(ys)
            elif etype == 'INSERT':
                ins = ent.dxf.insert
                ex = ins[0]
                ey = ins[1]
            elif etype == 'SPLINE':
                cps = ent.control_points
                if cps:
                    xs = [p[0] for p in cps]
                    ys = [p[1] for p in cps]
                    ex = sum(xs)/len(xs)
                    ey = sum(ys)/len(ys)
            elif etype == 'DIMENSION':
                dp = ent.dxf.defpoint
                ex = dp[0]
                ey = dp[1]
            elif etype == 'SOLID':
                vtxs = []
                if hasattr(ent.dxf, 'vtx0'):
                    vtxs.append(ent.dxf.vtx0)
                if hasattr(ent.dxf, 'vtx1'):
                    vtxs.append(ent.dxf.vtx1)
                if hasattr(ent.dxf, 'vtx2'):
                    vtxs.append(ent.dxf.vtx2)
                if hasattr(ent.dxf, 'vtx3'):
                    vtxs.append(ent.dxf.vtx3)
                if vtxs:
                    xs = [v.x for v in vtxs]
                    ys = [v.y for v in vtxs]
                    ex = sum(xs)/len(xs)
                    ey = sum(ys)/len(ys)
        except:
            pass

        q_ex = quantize_value(normalize_coordinate(ex, min_x, max_x))
        q_ey = quantize_value(normalize_coordinate(ey, min_y, max_y))

        # ParamA, ParamB
        pA = compute_paramA(etype, ent, doc, max_dim)
        pB = compute_paramB(etype, ent, doc, max_dim)
        # 量化
        # 对 DIMENSION => text_midpoint_x / y => 先做 normCoord => quant
        # 对 INSERT => scale_x / y => 先做 clip[-1,1] => quant
        # 对 MTEXT => char_height => length => norm->quant
        if etype == 'DIMENSION':
            # text_midpoint_x,y => 坐标 => normCoord
            pA_q = 0
            pB_q = 0
            if hasattr(ent.dxf,'text_midpoint'):
                mx, my, _ = ent.dxf.text_midpoint
                pA_q = quantize_value(normalize_coordinate(mx, min_x, max_x))
                pB_q = quantize_value(normalize_coordinate(my, min_y, max_y))
            q_pA = pA_q
            q_pB = pB_q
        elif etype == 'INSERT':
            # scale_x / scale_y => clip[-1,1] => quant
            sx_ = max(-1.0, min(1.0, (ent.dxf.xscale if hasattr(ent.dxf, 'xscale') else 1.0)))
            sy_ = max(-1.0, min(1.0, (ent.dxf.yscale if hasattr(ent.dxf, 'yscale') else 1.0)))
            q_pA = quantize_value(sx_)
            q_pB = quantize_value(sy_)
        elif etype == 'MTEXT':
            # char_height => length => norm->quant
            ch = ent.dxf.char_height if hasattr(ent.dxf, 'char_height') else 0
            q_pA = quantize_value(normalize_length(ch, max_dim))
            q_pB = 0
        else:
            q_pA = 0
            q_pB = 0

        # 汇总到 features[new_i, :]
        row = [0]*NUM_FEATURES
        row[FEAT_ETYPE]         = ecode
        row[FEAT_PATCH_ID]      = patch_id
        row[FEAT_FLAGS]         = flags_val
        row[FEAT_BBOX_CX]       = q_cx
        row[FEAT_BBOX_CY]       = q_cy
        row[FEAT_BBOX_W]        = q_bw
        row[FEAT_BBOX_H]        = q_bh
        row[FEAT_MAIN_ANGLE]    = q_angle1
        row[FEAT_SECOND_ANGLE]  = q_angle2
        row[FEAT_MAIN_LEN]      = q_main_len
        row[FEAT_ENTITY_X]      = q_ex
        row[FEAT_ENTITY_Y]      = q_ey
        row[FEAT_PARAMA]        = q_pA
        row[FEAT_PARAMB]        = q_pB

        features[new_i, :] = row

    # 6) 组装输出
    dxf_name = os.path.basename(dxf_path)
    result_dict = {
        "src": dxf_name,
        "n_num": n,
        "succs": new_succs,
        "features": features.tolist(),  # 14列
        "fname": dxf_name
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
# 7. 如果你需要批量处理某个目录下的 DXF，可添加示例
###############################################################################
if __name__ == "__main__":
    import glob

    input_dir = r"/home/vllm/encode/data/TRAIN"  # TODO: 修改为实际 DXF 目录
    output_dir = r"/home/vllm/encode/data/Geom/TRAIN_4096"          # TODO: 修改为输出 json 目录

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dxf_files = glob.glob(os.path.join(input_dir, "*.dxf"))
    for dxf_file in dxf_files:
        process_single_dxf(dxf_file, output_dir)
