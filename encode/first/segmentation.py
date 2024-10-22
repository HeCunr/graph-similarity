import glob
import math
from collections import defaultdict
import os
import ezdxf
import pandas as pd
from ezdxf.entities import BoundaryPathType, EdgeType

import change1

def distance(point1, point2, xscale=1.0, yscale=1.0):
    '''
    4参数或2参数
    :param point1:
    :param point2:
    :param xscale:
    :param yscale:
    :return:
    '''
    return math.sqrt(((point1[0] - point2[0])*xscale)**2 + ((point1[1] - point2[1])*yscale)**2)



def countAndRead_insert_types(file_path):
    doc = ezdxf.readfile(file_path)
    modelspace = doc.modelspace()

    entity_types = {}
    insert_types = []
    insert_count = {}
    i = 0
    for entity in modelspace:
        entity_type = entity.dxftype()
        if entity_type == 'INSERT':
            INSERTname = entity.dxf.name
            INSERTposition = entity.dxf.insert
            INSERTinfo = [INSERTname, INSERTposition, entity.dxf.xscale, entity.dxf.yscale, entity.dxf.rotation]
            insert_types.append(INSERTinfo)

        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

    for i in insert_types:
        print(i)
        insert_count[i[0]] = insert_count.get(i[0], 0) + 1
    print(insert_count)
    for item, count in insert_count.items():
        if count >= 3:
            print(f"{item}: {count}")


    df = pd.DataFrame.from_dict(entity_types, orient="index", columns=["Count"])
    print(df)

    return insert_types, insert_count


def get_block_info(file_path, blockname, insertsList):
    '''
    在文件路径中找到对应块名的块信息
    insertList结构:[INSERTname, INSERTposition, entity.dxf.xscale, entity.dxf.yscale, entity.dxf.rotation]
    '''
    doc = ezdxf.readfile(file_path)
    maxlength = 0
    xscale = 1
    yscale = 1
    for insert in insertsList:
        if insert['name'] == blockname:
            xscale = insert['xscale']
            yscale = insert['yscale']
    # 遍历DXF文档中的所有块定义
    for block in doc.blocks:
        if block.name == blockname:
            lines = []
            print(f"Block name: {block.name}")
            print(f"Block base point: {block.base_point}")

            #遍历块中的每个实体

            for entity in block:
                print(f"  Entity type: {entity.dxftype()}")
                # 你可以根据需要添加更多信息，例如实体的坐标、颜色等
                # 例如，如果是LINE实体，你可以这样打印起点和终点：
                if entity.dxftype() == 'LINE':

                    start_point = entity.dxf.start
                    end_point = entity.dxf.end
                    line_info = {
                        'start': (start_point[0], start_point[1]),
                        'end': (end_point[0], end_point[1])
                    }
                    lines.append(line_info)


                if entity.dxftype() == 'LWPOLYLINE':

                    vertices = entity.get_points('xy')
                    is_closed = entity.is_closed
                    # 如果LWPOLYLINE有多个顶点，我们可以创建多条LINE
                    print(is_closed)
                    for i in range(entity.dxf.count - 1):
                        start_point = vertices[i]
                        end_point = vertices[i + 1]

                        # 创建一个LINE实体信息字典
                        line_info = {
                            'start': (start_point[0], start_point[1]),
                            'end': (end_point[0], end_point[1])
                        }

                        # 将LINE实体信息添加到列表中
                        lines.append(line_info)
                        print(line_info)
                    if is_closed:
                        start_point = vertices[entity.dxf.count - 1]
                        end_point = vertices[0]
                        line_info = {
                            'start': (start_point[0], start_point[1]),
                            'end': (end_point[0], end_point[1])
                        }
                        lines.append(line_info)
                        print(line_info)

            counter = defaultdict(int)
            for line in lines:
                print(line)
                length = distance(line['start'], line['end'])
                counter[length] += 1
            sorted_numbers = sorted(counter.keys())  # 先对数字进行排序
            for number in sorted_numbers:
                count = counter[number]
                print(f"数字 {number} 出现了 {count} 次")

def ifInBlock(position, insertPosition, xscale=1.0, yscale=1.0):
    '''
    :param position: 点，几维没关系
    :param insertPosition: 点
    :return: 1，范围内；0，不在范围内
    '''
    x0 = 8.115473
    xm = 267.473570
    y0 = 9.029738
    ym = 179.643975
    x1 = 113.368534
    x2 = 66.722061
    y1 = 12.707386
    y2 = 21.200306
    x0 *= xscale
    x1 *= xscale
    x2 *= xscale
    xm *= xscale
    y0 *= yscale
    y1 *= yscale
    y2 *= yscale
    ym *= yscale


    flag = 0

    if position[0]> insertPosition[0]+x0 and position[0]< insertPosition[0]+x0+xm:
        if position[0]<insertPosition[0]+x0+x1:#x位于第一部分
            if position[1]>insertPosition[1]+y0 and position[1]<insertPosition[1]+y0+ym:
                flag = 1
        elif position[0]>=insertPosition[0]+x0+x1 and position[0]<insertPosition[0]+x0+x1+x2:
            if position[1]>insertPosition[1]+y0+y1 and position[1]<insertPosition[1]+y0+ym:
                flag = 1
        elif position[0]>=insertPosition[0]+x0+x1+x2 and position[0]<insertPosition[0]+x0+xm:
            if position[1]>insertPosition[1]+y0+y2 and position[1]<insertPosition[1]+y0+ym:
                flag = 1

    return flag

def deleteByBlocks(insertPosition,filepath,savename):

    xscale = 1.0
    yscale = 1.0
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    for entity in msp:
        if entity.dxftype() == 'INSERT':
            if entity.dxf.name == 'FR':
                xscale = entity.dxf.xscale
                yscale = entity.dxf.yscale
                break
    doc = ezdxf.readfile(filepath)
    modelspace = doc.modelspace()
    for entity in modelspace:
        if entity.dxftype() == 'LINE':
            position = entity.dxf.start

        elif entity.dxftype() == 'INSERT' or entity.dxftype() == 'TEXT' or entity.dxftype() == 'MTEXT':
            position = entity.dxf.insert

        elif entity.dxftype() == 'CIRCLE':
            position = entity.dxf.center

        elif entity.dxftype() == 'DIMENSION':
            position = entity.dxf.text_midpoint

        elif entity.dxftype() == 'LEADER':
            vertices = entity.vertices
            position = vertices[0]

        elif entity.dxftype() == 'ARC':
            position = entity.start_point

        elif entity.dxftype() == 'LWPOLYLINE':
            vertices = entity.get_points('xy')
            position = vertices[0]

        elif entity.dxftype() == 'HATCH':
            flag= 0
            position = (0, 0)
            for boundary_path in entity.paths:
                if flag ==1:
                    break
                # 遍历边界路径中的所有边
                #print(boundary_path.has_edge_paths)
                # print(boundary_path.type)
                if boundary_path.type is BoundaryPathType.EDGE:
                    #print(boundary_path.edges)
                    for edge in boundary_path.edges:#找到一个坐标即可break
                        if edge.type is EdgeType.LINE:
                            # print(edge.type)
                            # print(edge.start, edge.end)
                            position = edge.start
                            flag = 1
                            break
                        elif edge.type is EdgeType.ARC:
                            # print(edge.type)
                            # print(edge.center, edge.radius, edge.start_angle, edge.end_angle, edge.ccw)
                            position = edge.start_point
                            flag = 1
                            break

                        # else:
                            # print(edge.type)
                elif boundary_path.type is BoundaryPathType.POLYLINE:
                    # print(boundary_path.is_closed)
                    # print(boundary_path.vertices)
                    vertices = boundary_path.vertices
                    position = vertices[0]
                    flag = 1

        elif entity.dxftype() == 'SPLINE':
            vertices = entity.fit_points
            if vertices:
                position = vertices[0]  # 如果列表不为空，取出第一个元素
            else:
                position = (0,0)

        elif entity.dxftype() == 'ELLIPSE':
            position = entity.dxf.center

        if ifInBlock(position,insertPosition,xscale,yscale) == 0:#不在block中则删去
            entity.destroy()



    doc.saveas(savename)


def process_file(file_path):
    insertsList, insertCount = countAndRead_insert_types(file_path)
    fr_count = insertCount.get('FR', 0)  # 获取'FR'的数量，如果没有则为0

    base_name = os.path.splitext(os.path.basename(file_path))[0] + "_"
    for count in range(fr_count, 0, -1):  # 从'FR'的总数开始倒数到1
        aim = count
        for insert in insertsList:
            if insert[0] == 'FR':
                aim -= 1
                if aim == 0:
                    insertPosition = insert[1]
                    new_file_path = f"dataset\modified\{base_name}{count}.dxf"
                    print(new_file_path)

                    deleteByBlocks(insertPosition, file_path, new_file_path)


                    break  # 找到后退出循环，处理下一个count


def batch_process_files(folder_path):
    # 使用 glob 模块搜索指定文件夹下的所有 .dxf 文件
    dxf_files = glob.glob(os.path.join(folder_path, '**', '*.dxf'), recursive=True)


    # 遍历所有找到的 .dxf 文件并处理
    for file_path in dxf_files:
        print(f"processing{file_path}")
        process_file(file_path)


if __name__ == '__main__':
    # 假设您有一个包含多个文件路径的列表
    folder_path = '../../../Users/15653/dwg-cx/dataset'
    batch_process_files(folder_path)



