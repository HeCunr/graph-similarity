# -*- coding: utf-8 -*-
"""
merge.py

功能：
  1) 批量：遍历 "/home/vllm/encode/data/TRAIN" 目录下的所有DXF文件，两两组合，
     调用 merge_dxf() 进行合成，并输出到 "/home/vllm/encode/data/synData"。
  2) 单次合并：保留原始逻辑，可通过命令行 python merge.py dxf1_path dxf2_path output_path 进行单次合并。
"""

import sys
import random
import math
import os
from typing import List, Tuple
from itertools import combinations

import ezdxf
from ezdxf.entities import DXFEntity, BoundaryPathType, EdgeType

# 仅考虑以下12种类型
VALID_TYPES = [
    'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT', 'MTEXT',
    'HATCH', 'DIMENSION', 'LEADER', 'INSERT', 'SPLINE', 'SOLID'
]


def get_valid_entities(doc: ezdxf.document.Drawing):
    """
    从给定的 ezdxf Document 中获取所有有效实体(那12种).
    返回列表: [(entity, bounding_box), ...]
    其中 bounding_box = (minx, miny, maxx, maxy), 可能为 None 若无法计算
    """
    msp = doc.modelspace()
    valid_ents = []
    for e in msp:
        if e.dxftype() in VALID_TYPES:
            bbox = compute_entity_bounding_box(e, doc)
            valid_ents.append((e, bbox))
    return valid_ents


def compute_entity_bounding_box(entity: DXFEntity, doc: ezdxf.document.Drawing):
    """
    计算单个实体的外框 (minx, miny, maxx, maxy)。
    若无法计算则返回 None。
    """
    from math import sin, cos, radians

    etype = entity.dxftype()
    try:
        if etype == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            min_x = min(start[0], end[0])
            max_x = max(start[0], end[0])
            min_y = min(start[1], end[1])
            max_y = max(start[1], end[1])
            return (min_x, min_y, max_x, max_y)

        elif etype == 'CIRCLE':
            center = entity.dxf.center
            r = entity.dxf.radius
            return (center[0] - r, center[1] - r, center[0] + r, center[1] + r)

        elif etype == 'ARC':
            center = entity.dxf.center
            r = entity.dxf.radius
            start_a = radians(entity.dxf.start_angle)
            end_a = radians(entity.dxf.end_angle)
            if end_a < start_a:
                end_a += 2 * math.pi
            # 采样多点来近似包围盒
            n = 36
            angles = [start_a + i * (end_a - start_a) / (n - 1) for i in range(n)]
            xs = [center[0] + r * math.cos(a) for a in angles]
            ys = [center[1] + r * math.sin(a) for a in angles]
            return (min(xs), min(ys), max(xs), max(ys))

        elif etype == 'LWPOLYLINE':
            points = list(entity.get_points())
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (min(xs), min(ys), max(xs), max(ys))

        elif etype == 'TEXT':
            ip = entity.dxf.insert
            h = getattr(entity.dxf, 'height', 0)
            w = h  # 简化处理为高度 = 宽度
            return (ip[0], ip[1], ip[0] + w, ip[1] + h)

        elif etype == 'MTEXT':
            ip = entity.dxf.insert
            ch = getattr(entity.dxf, 'char_height', 0)
            w = getattr(entity.dxf, 'width', 0)
            return (ip[0], ip[1], ip[0] + w, ip[1] + ch)

        elif etype == 'HATCH':
            # 若需要准确，可以对边界进行采样
            return None

        elif etype == 'DIMENSION':
            dp = entity.dxf.defpoint
            if not dp:
                return None
            tx = entity.dxf.text_midpoint
            if not tx:
                tx = (dp[0], dp[1], 0)
            xs = [dp[0], tx[0]]
            ys = [dp[1], tx[1]]
            return (min(xs), min(ys), max(xs), max(ys))

        elif etype == 'LEADER':
            verts = entity.vertices
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            return (min(xs), min(ys), max(xs), max(ys))

        elif etype == 'INSERT':
            # 取块中所有实体的 bbox，做变换并合并
            insert_point = entity.dxf.insert
            xscale = getattr(entity.dxf, 'xscale', 1.0)
            yscale = getattr(entity.dxf, 'yscale', 1.0)
            rotation = getattr(entity.dxf, 'rotation', 0.0)
            block_name = entity.dxf.name
            try:
                block = doc.blocks.get(block_name)
            except:
                return None

            minxs, minys, maxxs, maxys = [], [], [], []
            for be in block:
                sub_bbox = compute_entity_bounding_box(be, doc)
                if sub_bbox:
                    (mnx, mny, mxx, mxy) = sub_bbox
                    # 只对 corners 做一次变换
                    corners = [(mnx, mny), (mnx, mxy), (mxx, mny), (mxx, mxy)]
                    ang = radians(rotation)
                    tx_corners = []
                    for (x0, y0) in corners:
                        # 缩放
                        x1, y1 = x0 * xscale, y0 * yscale
                        # 旋转
                        xr = x1 * math.cos(ang) - y1 * math.sin(ang)
                        yr = x1 * math.sin(ang) + y1 * math.cos(ang)
                        # 平移
                        x2 = xr + insert_point[0]
                        y2 = yr + insert_point[1]
                        tx_corners.append((x2, y2))
                    all_x = [p[0] for p in tx_corners]
                    all_y = [p[1] for p in tx_corners]
                    minxs.append(min(all_x))
                    maxxs.append(max(all_x))
                    minys.append(min(all_y))
                    maxys.append(max(all_y))
            if not minxs:
                return None
            return (min(minxs), min(minys), max(maxxs), max(maxys))

        elif etype == 'SPLINE':
            pts = entity.control_points if entity.control_points else []
            if not pts:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))

        elif etype == 'SOLID':
            v0 = (entity.dxf.vtx0.x, entity.dxf.vtx0.y)
            v1 = (entity.dxf.vtx1.x, entity.dxf.vtx1.y)
            v2 = (entity.dxf.vtx2.x, entity.dxf.vtx2.y)
            # v3 不一定存在
            v3 = v2
            if hasattr(entity.dxf, 'vtx3'):
                v3 = (entity.dxf.vtx3.x, entity.dxf.vtx3.y)
            xs = [v0[0], v1[0], v2[0], v3[0]]
            ys = [v0[1], v1[1], v2[1], v3[1]]
            return (min(xs), min(ys), max(xs), max(ys))

    except:
        return None

    return None


def get_bbox_of_entities(entities: List[Tuple[DXFEntity, Tuple[float, float, float, float]]]):
    """
    从若干 (entity, bbox) 中获取整体 bbox
    返回 (minx, miny, maxx, maxy)
    如果全是 None，就返回 (0,0,0,0)
    """
    xs, ys = [], []
    for e, box in entities:
        if box:
            xs.extend([box[0], box[2]])
            ys.extend([box[1], box[3]])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def offset_entity(entity: DXFEntity, offset_x: float, offset_y: float, doc: ezdxf.document.Drawing):
    """
    对实体进行平移操作，使其整体偏移(offset_x, offset_y).
    """
    etype = entity.dxftype()

    if etype == 'LINE':
        start = entity.dxf.start
        end = entity.dxf.end
        entity.dxf.start = (start[0] + offset_x, start[1] + offset_y)
        entity.dxf.end = (end[0] + offset_x, end[1] + offset_y)

    elif etype == 'CIRCLE':
        c = entity.dxf.center
        entity.dxf.center = (c[0] + offset_x, c[1] + offset_y)

    elif etype == 'ARC':
        c = entity.dxf.center
        entity.dxf.center = (c[0] + offset_x, c[1] + offset_y)

    elif etype == 'LWPOLYLINE':
        pts = entity.get_points()
        new_pts = []
        for p in pts:
            new_pts.append((p[0] + offset_x, p[1] + offset_y, p[2]))  # (x, y, bulge)
        entity.set_points(new_pts)

    elif etype in ['TEXT', 'MTEXT']:
        ip = entity.dxf.insert
        entity.dxf.insert = (ip[0] + offset_x, ip[1] + offset_y)

    elif etype == 'DIMENSION':
        dp = entity.dxf.defpoint
        if dp:
            entity.dxf.defpoint = (dp[0] + offset_x, dp[1] + offset_y, dp[2])
        tm = entity.dxf.text_midpoint
        if tm:
            entity.dxf.text_midpoint = (tm[0] + offset_x, tm[1] + offset_y, tm[2])

    elif etype == 'LEADER':
        verts = entity.vertices
        new_verts = []
        for v in verts:
            new_verts.append((v[0] + offset_x, v[1] + offset_y))
        entity.vertices = new_verts

    elif etype == 'INSERT':
        ins = entity.dxf.insert
        entity.dxf.insert = (ins[0] + offset_x, ins[1] + offset_y)

    elif etype == 'SPLINE':
        cps = list(entity.control_points)
        new_cps = []
        for cp in cps:
            new_cps.append((cp[0] + offset_x, cp[1] + offset_y, cp[2]))
        entity.control_points = new_cps

    elif etype == 'SOLID':
        v0 = entity.dxf.vtx0
        v1 = entity.dxf.vtx1
        v2 = entity.dxf.vtx2
        if hasattr(entity.dxf, 'vtx3'):
            v3 = entity.dxf.vtx3
            entity.dxf.vtx3 = (v3.x + offset_x, v3.y + offset_y, v3.z)
        entity.dxf.vtx0 = (v0.x + offset_x, v0.y + offset_y, v0.z)
        entity.dxf.vtx1 = (v1.x + offset_x, v1.y + offset_y, v1.z)
        entity.dxf.vtx2 = (v2.x + offset_x, v2.y + offset_y, v2.z)

    elif etype == 'HATCH':
        # 通过 boundary_path 的类型和 edges 来偏移各顶点
        for boundary_path in entity.paths:
            if boundary_path.type == BoundaryPathType.POLYLINE:
                new_vertices = []
                for vertex in boundary_path.vertices:
                    x, y, *rest = vertex
                    x += offset_x
                    y += offset_y
                    new_vertices.append((x, y, *rest))
                boundary_path.vertices = new_vertices

            elif boundary_path.type == BoundaryPathType.EDGE:
                for edge in boundary_path.edges:
                    # 直线边
                    if edge.type == EdgeType.LINE:
                        s = edge.start
                        e = edge.end
                        edge.start = (s[0] + offset_x, s[1] + offset_y)
                        edge.end = (e[0] + offset_x, e[1] + offset_y)

                    # 圆弧或椭圆
                    elif edge.type in [EdgeType.ARC, EdgeType.ELLIPSE]:
                        c = edge.center
                        edge.center = (c[0] + offset_x, c[1] + offset_y)


def merge_dxf(dxf_path1, dxf_path2, output_path):
    """
    合并核心逻辑：
      1) 打开两个DXF文件，收集12种类型实体以及其外包盒
      2) 随机确定合并后实体总数 M (在二者数量之间)
      3) 从两边的实体中随机抽取一些，总计 M 个
      4) 将抽取到的一部分实体“平移”到相对不冲突的位置（示例为简单左右平移）
      5) 写入到新的DXF文件
    """
    doc1 = ezdxf.readfile(dxf_path1)
    doc2 = ezdxf.readfile(dxf_path2)

    ents1 = get_valid_entities(doc1)
    ents2 = get_valid_entities(doc2)

    n1 = len(ents1)
    n2 = len(ents2)

    if n1 == 0 and n2 == 0:
        print(f"[警告] 两个文件都没有可用实体：{dxf_path1} 和 {dxf_path2}")
        return

    min_n = min(n1, n2)
    max_n = max(n1, n2)
    if min_n == max_n:
        # 二者数量相同时的特殊处理
        if min_n > 1:
            M = random.randint(min_n - 1, min_n)  # 随机让数量少一点
        else:
            M = min_n
    else:
        # 在 (min_n, max_n) 之间选 M
        if (max_n - min_n) > 1:
            M = random.randint(min_n + 1, max_n - 1)
        else:
            M = min_n

    # 分配 M
    ratio = random.random()  # 0~1
    c1 = int(round(M * ratio))
    c2 = M - c1
    c1 = min(c1, n1)
    c2 = min(c2, n2)
    if c1 + c2 < M:
        c1 = min(n1, c1 + (M - (c1 + c2)))
        c2 = min(n2, M - c1)

    selected1 = random.sample(ents1, c1) if c1 > 0 else []
    selected2 = random.sample(ents2, c2) if c2 > 0 else []

    # 分别算两边选中实体的整体 bbox
    box1 = get_bbox_of_entities(selected1)
    box2 = get_bbox_of_entities(selected2)

    # 新的 DXF 文档
    new_doc = ezdxf.new('R2018')
    new_msp = new_doc.modelspace()

    # 把 selected1 加入到 new_msp
    copies1 = []
    for (ent, bbox) in selected1:
        new_ent = ent.copy()  # 复制实体
        new_msp.add_entity(new_ent)
        copies1.append(new_ent)

    # 再把 selected2 加入到 new_msp
    copies2 = []
    for (ent, bbox) in selected2:
        new_ent = ent.copy()
        new_msp.add_entity(new_ent)
        copies2.append(new_ent)

    (min1x, min1y, max1x, max1y) = box1
    (min2x, min2y, max2x, max2y) = box2

    width1 = max1x - min1x if (max1x > min1x) else 0
    width2 = max2x - min2x if (max2x > min2x) else 0

    # 简单地把第二组整体平移到第一组右边
    spacing = width1 * 0.1 if width1 > 0 else 10.0
    offset_x = (max1x + spacing) - min2x

    # 对 copies2 做平移
    if c1 > 0 and c2 > 0:
        for ent in copies2:
            offset_entity(ent, offset_x, 0.0, new_doc)

    # 保存
    new_doc.saveas(output_path)
    print(f"[合成完成] 实体数 M={M}，输出：{output_path}")


def batch_merge_dxf():
    """
    遍历 "/home/vllm/encode/data/TRAIN" 目录下所有 dxf 文件，两两合成，输出至 "/home/vllm/encode/data/synData"
    """
    train_path = "/home/vllm/encode/data/TRAIN"
    syn_path = "/home/vllm/encode/data/synData"

    # 如果输出目录不存在，则创建
    if not os.path.exists(syn_path):
        os.makedirs(syn_path)

    # 获取目录下所有 dxf 文件
    dxf_files = [
        f for f in os.listdir(train_path)
        if f.lower().endswith('.dxf')
    ]
    dxf_files.sort()

    # 计算两两组合并进行处理
    for i, j in combinations(range(len(dxf_files)), 2):
        f1 = os.path.join(train_path, dxf_files[i])
        f2 = os.path.join(train_path, dxf_files[j])
        # 可以自定义输出的文件名形式
        out_name = f"{os.path.splitext(dxf_files[i])[0]}_" \
                   f"{os.path.splitext(dxf_files[j])[0]}.dxf"
        out_path = os.path.join(syn_path, out_name)

        print(f"正在合成: {f1} + {f2} -> {out_path}")
        try:
            merge_dxf(f1, f2, out_path)
        except Exception as e:
            print(f"[错误] 合成 {f1} 和 {f2} 时出现问题: {e}")


def main():
    """
    如果直接执行 python merge.py 不加参数，则进行批量两两合成；
    如果执行 python merge.py dxf1 dxf2 out 则进行单次合成。
    """
    if len(sys.argv) == 1:
        # 不带参数，进行批量合成
        batch_merge_dxf()
    elif len(sys.argv) == 4:
        # 带3个参数，单次合成
        dxf1 = sys.argv[1]
        dxf2 = sys.argv[2]
        out_path = sys.argv[3]
        merge_dxf(dxf1, dxf2, out_path)
    else:
        print("用法：")
        print("  1) 批量模式：python merge.py")
        print("  2) 单次模式：python merge.py <dxf1> <dxf2> <out_dxf>")


if __name__ == '__main__':
    main()
