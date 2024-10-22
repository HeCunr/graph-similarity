import ezdxf
import ezdxf.entities
from ezdxf.entities import BoundaryPathType, EdgeType
import segmentation
import count

def extract_arc_info_and_save_to_list(filename):
    # 初始化一个空列表来保存ARC信息
    arc_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "ARC":
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            start_point = entity.start_point
            end_point = entity.end_point
            # 将ARC信息保存为一个字典，并添加到列表中
            arc_info = {
                "center": center,
                "radius": radius,
                "start_angle": start_angle,
                "end_angle": end_angle,
                "start_point": start_point,
                "end_point": end_point
            }
            arc_info_list.append(arc_info)

            # 输出ARC信息列表
    print(arc_info_list)
    return arc_info_list

def extract_line_info_and_save_to_list(filename):
    line_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            line_info = {
                "start": start,
                "end": end
            }
            line_info_list.append(line_info)

    return line_info_list

def extract_circle_info_and_save_to_list(filename):
    circle_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "CIRCLE":
            center = entity.dxf.center
            radius = entity.dxf.radius
            circle_info = {
                "center": center,
                "radius": radius
            }
            circle_info_list.append(circle_info)

    return circle_info_list

def extract_polyline_info_and_save_to_list(filename):
    polyline_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "POLYLINE":
            # 提取POLYLINE实体的一组顶点
            points = entity.dxf.points()
            polyline_info = {
                "points": points
            }
            polyline_info_list.append(polyline_info)

    return polyline_info_list

def lwpolyline_to_lines(dxf_file):
    # 存储LINE实体信息的列表
    lines = []

    # 加载DXF文档
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    # 遍历所有的LWPOLYLINE实体
    for lwpolyline in msp.query('LWPOLYLINE'):
        # 获取LWPOLYLINE的顶点，并确保它是一个列表
        vertices = lwpolyline.get_points('xy')
        is_closed = lwpolyline.is_closed
        # 如果LWPOLYLINE有多个顶点，我们可以创建多条LINE
        print(is_closed)
        for i in range(lwpolyline.dxf.count - 1):
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
            start_point = vertices[lwpolyline.dxf.count-1]
            end_point = vertices[0]
            line_info = {
                'start': (start_point[0], start_point[1]),
                'end': (end_point[0], end_point[1])
            }
            lines.append(line_info)
            print(line_info)

    return lines

def extract_lwpolyline_info_and_save_to_list(filename):
    lwpolyline_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "LWPOLYLINE":
            # 提取LWPOLYLINE实体的一组顶点
            points = entity.get_points('xy')  #xyseb作为参数，字符串，决定读取的内容
            lwpolyline_info = {
                "points": points
            }
            lwpolyline_info_list.append(lwpolyline_info)

    return lwpolyline_info_list

def extract_text_info_and_save_to_list(filename):
    text_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "TEXT":
            insert = entity.dxf.insert  # 提取TEXT实体的插入点位置和旋转角度等属性
            text_info = {
                "insert": insert,  # 这里仅提取插入点作为示例，实际应用中可能需要提取其他属性。
                "text": entity.dxf.text  # 提取TEXT实体中的文本内容
            }
            text_info_list.append(text_info)

    return text_info_list

def extract_mtext_info_and_save_to_list(filename):
    mtext_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "MTEXT":
            insert = entity.dxf.insert  # 提取MTEXT实体的插入点位置和旋转角度等属性
            mtext_info = {
                "insert": insert,  # 这里仅提取插入点作为示例，实际应用中可能需要提取其他属性。
                "text": entity.dxf.text  # 提取MTEXT实体中的文本内容
            }
            mtext_info_list.append(mtext_info)

    return mtext_info_list

def extract_dimension_info_and_save_to_list(filename):
    dimension_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "DIMENSION":
            # 提取DIMENSION实体的相关信息，这里仅提取位置作为示例
            text = entity.dxf.text
            midpoint = entity.dxf.text_midpoint
            text_rotation = entity.dxf.text_rotation
            dimension_info = {
                "text": text,
                "midpoint": midpoint,
                "text_rotation": text_rotation
            }
            dimension_info_list.append(dimension_info)
    print(dimension_info_list)
    return dimension_info_list

def extract_insert_info_and_save_to_list(filename):
    insert_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "INSERT":
            # 提取INSERT实体的相关信息，这里仅提取位置和缩放因子作为示例
            name = entity.dxf.name
            insert = entity.dxf.insert
            xscale = entity.dxf.xscale
            yscale = entity.dxf.yscale
            rotation = entity.dxf.rotation

            insert_info = {
                "name": name,
                "insert": insert,
                "xscale": xscale,
                "yscale": yscale,
                "rotation": rotation
            }
            insert_info_list.append(insert_info)
    print(insert_info_list)
    return insert_info_list

def extract_leader_info_and_save_to_list(filename):
    leader_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "LEADER":
            # 提取LEADER实体的相关信息，这里仅提取起点和终点作为示例
            vertices = entity.vertices

            leader_info = {
                "vertices": vertices,

            }
            leader_info_list.append(leader_info)
    print(leader_info_list)
    return leader_info_list

def extract_solid_info_and_save_to_list(filename):
    solid_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "SOLID":
            vertices = entity.vertices()

            solid_info_list.append(vertices)

    return solid_info_list

def extract_hatch_info_and_save_to_list(filename):
    hatch_info_list = []

    doc = ezdxf.readfile(filename)
    # layer_count = len(doc.layers)
    # print(doc.layers)
    # print(f"The DXF file contains {layer_count} layers.")
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == "HATCH":

            pattern_name = entity.dxf.pattern_name
            solid_fill = entity.dxf.solid_fill
            hatch_style = entity.dxf.hatch_style
            pattern_angle = entity.dxf.pattern_angle
            hatch_style = entity.dxf.hatch_style
            # print("seeds")
            # print(entity.seeds)

            print(entity.dxf.__dict__)
            # 遍历所有的边界路径
            # for seeds in entity.seeds:
            #     print(seeds)
            for boundary_path in entity.paths:
                # 遍历边界路径中的所有边
                #print(boundary_path.has_edge_paths)
                print(boundary_path.type)
                if boundary_path.type is BoundaryPathType.EDGE:
                    #print(boundary_path.edges)
                    for edge in boundary_path.edges:
                        if edge.type is EdgeType.LINE:
                            print(edge.type)
                            print(edge.start, edge.end)
                        elif edge.type is EdgeType.ARC:
                            print(edge.type)
                            print(edge.center, edge.radius, edge.start_angle, edge.end_angle, edge.ccw)
                        elif edge.type is EdgeType.ELLIPSE:
                            print(edge.type)
                            print(edge.major_axis_vector, edge.minor_axis_length, edge.start_angle, edge.end_angle, edge.ccw)
                        else:
                            print(edge.type)
                elif boundary_path.type is BoundaryPathType.POLYLINE:
                    print(boundary_path.is_closed)
                    print(boundary_path.vertices)
                    #print(boundary_path.source_boundary_objects)

def extract_spline_info_and_save_to_list(filename):
    solid_info_list = []

    doc = ezdxf.readfile(filename)
    modelspace = doc.modelspace()

    for entity in modelspace:
        if entity.dxftype() == 'SPLINE':
            vertices = entity.fit_points
            print(vertices)




filename = 'dataset/Job-1306 (1)/QFN19LA(Cu) -502 Rev1.dxf'
import time
#测试语句
if __name__ == '__main__':
    start_time = time.perf_counter()

    # count.count_entity_types(filename)
    # lineList = extract_line_info_and_save_to_list(filename)      #ok
    #
    # arcList = extract_arc_info_and_save_to_list(filename)      #ok
    # circleList = extract_circle_info_and_save_to_list(filename)     #ok
    # polyLineList = extract_polyline_info_and_save_to_list(filename)     #无
    # lwpolyLineList = extract_lwpolyline_info_and_save_to_list(filename)    #ok
    # lwpolyLineList_to_lines = lwpolyline_to_lines(filename)  #ok
    # textList = extract_text_info_and_save_to_list(filename)     #ok
    # mtextList = extract_mtext_info_and_save_to_list(filename)     #ok
    # dimensionList = extract_dimension_info_and_save_to_list(filename)     #ok
    insertList = extract_insert_info_and_save_to_list(filename)      #ok
    segmentation.get_block_info(filename, 'FR', insertList)
    # leaderList = extract_leader_info_and_save_to_list(filename)   #ok
    # hatchList = extract_hatch_info_and_save_to_list(filename)     #边缘路径情况很复杂，单独考虑
    # solidList = extract_solid_info_and_save_to_list(filename)
    extract_spline_info_and_save_to_list(filename)
    # for list in [ arcList, solidList ]:
    #     # print(len(list))
    #     print(list)
    end_time = time.perf_counter()
    print("程序运行总时长："+str(end_time - start_time)+ "秒")



# lines_info = lwpolyline_to_lines(filename)
#
# # 打印结果
# for line in lines_info:
#     print(f'LINE from ({line["start"][0]}, {line["start"][1]}) to ({line["end"][0]}, {line["end"][1]})')
