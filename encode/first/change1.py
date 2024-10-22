import glob
import re
from html import entities

import get_info
import ezdxf
import math
import os
import time
# os.remove('gray1.dxf')
def processed_lines(lines):
    processed_lines = []

    for line in lines:
        x1 = float(line['start'][0])
        y1 = float(line['start'][1])
        x2 = float(line['end'][0])
        y2 = float(line['end'][1])


        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = 0
        if x1!=x2 or y1!=y2:
            angle = abs((y2 - y1)) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # print(angle)
        processed_line = [x, y, length,  angle]
        processed_lines.append(processed_line)

    return processed_lines

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_centroid(vertices):
    # 确保顶点列表至少包含三个点
    if len(vertices) < 3:
        raise ValueError("A polygon must have at least 3 vertices.")

        # 初始化质心坐标的累加变量
    centroid_x = 0
    centroid_y = 0

    # 遍历所有顶点并累加坐标
    for vertex in vertices:
        centroid_x += vertex[0]
        centroid_y += vertex[1]

        # 计算坐标的平均值以得到质心
    centroid_x /= len(vertices)
    centroid_y /= len(vertices)

    # 返回质心坐标
    return centroid_x, centroid_y


def processed_solids(solids):
    processed_solids = []
    for solid in solids:
        centroid_x, centroid_y =calculate_centroid(solid)
        processed_solid = [centroid_x, centroid_y]
        processed_solids.append(processed_solid)
    return processed_solids

def processed_circles(circles):
    processed_circles = []

    for circle in circles:
        cx = float(circle['center'][0])
        cy = float(circle['center'][1])
        r = float(circle['radius'])

        processed_circle = [cx, cy, r]
        processed_circles.append(processed_circle)
    return processed_circles

def processed_arcs(arcs):
    processed_arcs = []

    for arc in arcs:
        cx = (float(arc['start_point'][0])+float(arc['start_point'][0]))/2
        cy = (float(arc['start_point'][1])+float(arc['start_point'][1]))/2
        r = float(arc['radius'])
        start_angle = float(arc['start_angle'])
        end_angle = float(arc['end_angle'])


        processed_arc = [cx, cy, r, start_angle, end_angle]
        processed_arcs.append(processed_arc)
    return processed_arcs

def compare_one_circle(circle1,circles2):
        mvalue = 0
        mp = 0
        ml = 0
        mindis = 100
        i = 0
        p=0
        l=0
        value=0
        flag = 0  # 锁定选定的位置
        for circle2 in circles2:  # 一条一条对比距离
            dis1 = math.sqrt((circle2[1] - circle1[1]) ** 2 + (circle2[0] - circle1[0]) ** 2)
            p = (6 * circle1[2] - dis1) / (6 * circle1[2])
            if p < 0:  # <0的处理
                p = 0
            print('p' + str(p))

            l = (max(circle1[2], circle2[2]) - abs(circle1[2] - circle2[2])) / (max(circle1[2], circle2[2]))
            if l < 0:
                l = 0
            print('l' + str(l))

            value = l*0.5+p*0.5
            if value >=0.98:
                mvalue = value
                mp = p
                ml = l
                break

            if value > mvalue:
                mvalue = value
                mp = p
                ml = l

        return mvalue,mp,ml

def compare_one_arc(arc1,arcs2):
    dis1 = 0
    mvalue = 0
    mp = 0
    ml = 0
    ma = 0
    for arc2 in arcs2:
        dis1 = math.sqrt((arc2[1] - arc1[1]) ** 2 + (arc2[0] - arc1[0]) ** 2)
        p = (18 * arc1[2] - dis1) / (18 * arc1[2])
        if p < 0:  # <0的处理
            p = 0
        l = (max(arc1[2], arc2[2]) - abs(arc1[2] - arc2[2])) / (max(arc1[2], arc2[2]))
        if l < 0:
            l = 0
        arccenter1 = (arc1[4] + arc1[3]) / 2
        arccenter2 = (arc2[4] + arc2[3]) / 2
        alength1 = arc1[4] - arc1[3]
        alength2 = arc2[4] - arc2[3]
        a = (360 - abs((arccenter1 - arccenter2))) / 720 + (360 - abs(alength1 - alength2) ) / 720
        value = l * 0.5 + p * 0.1 + a * 0.4

        if value >=0.92:
            mvalue = value
            mp = p
            ml = l
            ma = a
            break

        if value > mvalue:
            mvalue = value
            mp = p
            ml = l
            ma = a
    return mvalue, mp, ml, ma




def compare_one_text(text1, texts2):
    mindis = 1
    # i = 0
    flag = 0  # 锁定选定的位置
    for text2 in texts2:  # 一条一条对比距离
        dis1 = math.sqrt((text1['insert'][0] - text2['insert'][0]) ** 2 + (text1['insert'][1] - text2['insert'][1]) ** 2)
        if text1['text'] == text2['text'] and dis1 < mindis:
            flag = 1
            break

    print('textflag' + str(flag))
    return flag

def compare_one_solid(solid1,solids2):
    mvalue = 0
    for solid2 in solids2:
        dis1 = distance(solid1, solid2)
        p = (0.5-dis1)/0.5
        value = p
        if value >= 0.98:
            mvalue = value
            break
        if value > mvalue:
            mvalue = value
    return mvalue


def compare_one_line(line1,lines2):

        mindis = 100
        flag = 0 #锁定选定的位置
        mp=0
        ml=0
        ma=0
        mvalue = 0
        for line2 in lines2:#一条一条对比距离

            dis1 = math.sqrt((line2[1]-line1[1])**2+(line2[0]-line1[0])**2)
            p = 1
            if line1[2] != 0:
                p = (6 * line1[2] - dis1) / (6 * line1[2])
            if p < 0:  # <0的处理
                p = 0

            l = 1
            if line1[2] != 0 and line2[2] != 0:
                l = (1 * max(line1[2], line2[2]) - abs(line1[2] - line2[2])) / (1 * max(line1[2], line2[2]))
            if l < 0:
                l = 0

            a = 1 - abs(line1[3] - line2[3])

            value = (p * 0.2 + l * 0.4 + a * 0.4)

            if value >= 0.98:
                mvalue = value
                mp = p
                ml = l
                ma = a
                break

            if value > mvalue:
                mvalue = value
                mp = p
                ml = l
                ma = a

        return mvalue, mp, ml, ma







def dxfAdxf(dxf1,dxf2):
    input1 = extract_filename_without_extension(dxf1)#仅提取文件名，去除路径和扩展名
    input2 = extract_filename_without_extension(dxf2)
    start_time = time.perf_counter()
    doc = ezdxf.readfile(dxf1)
    modelspace = doc.modelspace()
    print('Reset entities')
    for entity in modelspace:
        entity.dxf.color = 8
    print('Reset layers')
    for layer in doc.layers:
        layer.dxf.color = 8

    doc.saveas('gray2.dxf')
    print('Reading dxf2...')
    print('Reading Lines...')
    linelist2 = get_info.extract_line_info_and_save_to_list(dxf2)
    print('Reading Circles...')
    circlelist2 = get_info.extract_circle_info_and_save_to_list(dxf2)
    print('Reading Texts...')
    texts2 = get_info.extract_text_info_and_save_to_list(dxf2)
    print('Reading Arcs...')
    arcs2 = get_info.extract_arc_info_and_save_to_list(dxf2)
    print('Reading Solids...')
    solids2 = get_info.extract_solid_info_and_save_to_list(dxf2)
    print(solids2)
    print('Processing...')
    parcs2 = processed_arcs(arcs2)
    psolids2 = processed_solids(solids2)
    print(psolids2)
    plines2 = processed_lines(linelist2)

    pcircles2 = processed_circles(circlelist2)

    doc = ezdxf.readfile('gray2.dxf')

    modelspace = doc.modelspace()
    print('Comparing CIRCLES')
    for entity in modelspace:
        if entity.dxftype() == 'CIRCLE':
            cx = entity.dxf.center[0]
            cy = entity.dxf.center[1]
            radius = entity.dxf.radius
            thisCircle = [cx, cy, radius]

            value, p, l = compare_one_circle(thisCircle, pcircles2)
            if value < 0.98:
                if p <= 0.9 and l > 0.9:
                    entity.dxf.color = 7
                else:
                    entity.dxf.color = 1
    count1 = 0

    print('Comparing LINES')
    for entity in modelspace:
        if entity.dxftype() == 'LINE':
            count1 += 1
            x1 = entity.dxf.start[0]
            y1 = entity.dxf.start[1]
            x2 = entity.dxf.end[0]
            y2 = entity.dxf.end[1]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = 0
            if x1 != x2 or y1 != y2:
                angle = abs((y2 - y1)) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # print(angle)
            processed_line = [x, y, length, angle]

            value,p,l,a = compare_one_line(processed_line, plines2)
            if value < 0.95:
                if (p<=0.9 and l>0.9 and a >0.9) or (a > 0.98):
                    entity.dxf.color = 7
                else:
                    entity.dxf.color = 1
            print(value)
            print(count1)

    print('Comparing TEXTS')
    for entity in modelspace:
        if entity.dxftype() == 'TEXT':
            insert = entity.dxf.insert  # 提取TEXT实体的插入点位置和旋转角度等属性
            text1 = {
                "insert": insert,  # 这里仅提取插入点作为示例，实际应用中可能需要提取其他属性。
                "text": entity.dxf.text  # 提取TEXT实体中的文本内容
            }
            flag1 = compare_one_text(text1, texts2)
            if flag1 != 1:
                entity.dxf.color = 1
    end_time = time.perf_counter()

    i = 0
    print('Comparing ARCS')
    for entity in modelspace:
        if entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            start_point = entity.start_point
            end_point = entity.end_point
            # 将ARC信息保存为一个字典，并添加到列表中
            arc = {
                "center": center,
                "radius": radius,
                "start_angle": start_angle,
                "end_angle": end_angle,
                "start_point": start_point,
                "end_point": end_point
            }
            cx = (float(arc['start_point'][0]) + float(arc['start_point'][0])) / 2
            cy = (float(arc['start_point'][1]) + float(arc['start_point'][1])) / 2
            r = float(arc['radius'])
            start_angle = float(arc['start_angle'])
            end_angle = float(arc['end_angle'])

            arc1 = [cx, cy, r, start_angle, end_angle]
            value, p, l, a = compare_one_arc(arc1, parcs2)
            print(p, l, a)
            if value < 0.92:
                entity.dxf.color = 2
            i += 1
            print(i)
            print(value)

    print("Comparing SOLIDS")
    i = 0
    for entity in modelspace:
        if entity.dxftype() == "SOLID":
            solid = entity.vertices()
            centroid_x, centroid_y = calculate_centroid(solid)
            solid1 = [centroid_x, centroid_y]

            value = compare_one_solid(solid1, psolids2)
            if value < 0.98:
                entity.dxf.color = 1
            i += 1
            print(i)
            print(value)

    print('Time: ', end_time - start_time)
    doc.saveas(f"dataset/test/{input1}__{input2}.dxf")





def extract_filename_without_extension(input_string):
    # 正则表达式同时考虑正斜杠（/）和反斜杠（\）作为路径分隔符
    match = re.search(r'[/\\]([^/\\]+)\.dxf$', input_string)
    if match:
        return match.group(1)
    else:
        return None

    # 测试函数




if __name__ == '__main__':
    folder_path = 'dataset/test'
    dxf2 = "dataset/QFN19LB(Cu) -503 Rev1.dxf"
    file_paths = glob.glob(os.path.join(folder_path, '**', '*.dxf'), recursive=True)
    print(file_paths)
    for file in file_paths:
        dxf1 = file
        dxfAdxf(dxf1,dxf2)



