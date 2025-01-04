import ezdxf
from ezdxf.math import BoundingBox
import math
import os

def print_dxf_info(doc):
    """打印DXF文件的详细信息"""
    print("\nDXF文件信息:")
    print(f"DXF版本: {doc.dxfversion}")
    print(f"编码: {doc.encoding}")

    # 打印所有图层
    print("\n图层列表:")
    for layer in doc.layers:
        print(f"- {layer.dxf.name}")

    # 打印所有块定义
    print("\n块定义列表:")
    for block in doc.blocks:
        print(f"- 块名: {block.name}")
        # 打印块中的实体
        entity_types = {}
        for entity in block:
            entity_type = entity.dxftype()
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        if entity_types:
            print(f"  包含的实体类型:")
            for entity_type, count in entity_types.items():
                print(f"    * {entity_type}: {count}个")

    # 打印模型空间中的实体
    print("\n模型空间实体:")
    entity_types = {}
    for entity in doc.modelspace():
        entity_type = entity.dxftype()
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    for entity_type, count in entity_types.items():
        print(f"- {entity_type}: {count}个")

def get_tablet_bounds_and_refs(dxf_path):
    """获取所有TABLET块的边界坐标和引用对象"""
    doc = ezdxf.readfile(dxf_path)

    # 打印DXF文件详细信息
    print_dxf_info(doc)

    msp = doc.modelspace()
    tablet_data = []

    # 尝试不同的方式查询块引用
    print("\n尝试查找TABLET块引用...")

    # 方法1：直接查询所有INSERT实体
    all_inserts = list(msp.query('INSERT'))
    print(f"方法1 - 直接查询INSERT实体数量: {len(all_inserts)}")

    # 方法2：遍历所有实体查找INSERT
    insert_count = 0
    for entity in msp:
        if entity.dxftype() == 'INSERT':
            insert_count += 1
            print(f"找到块引用: {entity.dxf.name}")
    print(f"方法2 - 遍历实体找到INSERT数量: {insert_count}")

    # 方法3：特别查找TABLET
    tablet_pattern = "TABLET"
    for entity in msp:
        if entity.dxftype() == 'INSERT' and entity.dxf.name.upper() == tablet_pattern:
            print(f"\n找到TABLET块引用:")
            print(f"- 块名称: {entity.dxf.name}")
            print(f"- 插入点: {entity.dxf.insert}")
            print(f"- 缩放: ({entity.dxf.xscale}, {entity.dxf.yscale})")
            print(f"- 旋转: {entity.dxf.rotation}度")

            try:
                block_def = doc.blocks[entity.dxf.name]
                bbox = BoundingBox()
                for block_entity in block_def:
                    bbox.extend(block_entity.get_bbox())

                corners = []
                min_x, min_y, _ = bbox.extmin
                max_x, max_y, _ = bbox.extmax
                corners_local = [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y)
                ]

                # 应用变换
                rotation_rad = math.radians(entity.dxf.rotation)
                insertion_point = entity.dxf.insert
                scale_x = entity.dxf.xscale
                scale_y = entity.dxf.yscale

                transformed_corners = []
                for x, y in corners_local:
                    # 缩放
                    x_scaled = x * scale_x
                    y_scaled = y * scale_y

                    # 旋转
                    x_rot = x_scaled * math.cos(rotation_rad) - y_scaled * math.sin(rotation_rad)
                    y_rot = x_scaled * math.sin(rotation_rad) + y_scaled * math.cos(rotation_rad)

                    # 平移
                    x_final = x_rot + insertion_point[0]
                    y_final = y_rot + insertion_point[1]

                    transformed_corners.append((x_final, y_final))

                tablet_data.append({
                    'bounds': transformed_corners,
                    'handle': entity.dxf.handle
                })

            except Exception as e:
                print(f"处理TABLET块时出错: {str(e)}")

    print(f"\n总共找到 {len(tablet_data)} 个TABLET块引用")
    return tablet_data

def is_point_in_polygon(point, polygon):
    """判断点是否在多边形内部"""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                 (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i

    return inside

def get_entity_insertion_point(entity):
    """获取实体的插入点"""
    if entity.dxftype() == 'INSERT':
        # 对于块引用，直接返回其插入点
        return (entity.dxf.insert[0], entity.dxf.insert[1])
    else:
        try:
            # 对于其他实体，获取其几何中心
            bbox = entity.get_bbox()
            center_x = (bbox.extmin[0] + bbox.extmax[0]) / 2
            center_y = (bbox.extmin[1] + bbox.extmax[1]) / 2
            return (center_x, center_y)
        except:
            # 如果无法获取边界框，返回None
            return None

def extract_entities_for_tablet(doc, tablet_data, output_path_template):
    """为每个TABLET区域提取实体并创建新的DXF文件"""
    msp = doc.modelspace()

    # 遍历每个TABLET区域
    for index, tablet in enumerate(tablet_data):
        # 创建新的DXF文档
        new_doc = ezdxf.new('R2018')
        new_msp = new_doc.modelspace()

        # 复制所有需要的块定义
        copied_block_names = set()  # 用于追踪已复制的块

        # 遍历所有实体
        for entity in msp:
            # 跳过TABLET块
            if entity.dxftype() == 'INSERT' and entity.dxf.name == 'TABLET':
                continue

            # 获取实体的插入点
            insertion_point = get_entity_insertion_point(entity)
            if insertion_point is None:
                continue

            # 检查是否在当前TABLET边界内
            if is_point_in_polygon(insertion_point, tablet['bounds']):
                # 如果是块引用，确保块定义被复制
                if entity.dxftype() == 'INSERT':
                    block_name = entity.dxf.name
                    if block_name not in copied_block_names:
                        # 复制块定义
                        if block_name in doc.blocks and block_name.lower() not in ('*model_space', '*paper_space', '*paper_space0'):
                            block_def = doc.blocks[block_name]
                            new_block = new_doc.blocks.new(name=block_name)
                            for block_entity in block_def:
                                new_block.add_entity(block_entity.copy())
                            copied_block_names.add(block_name)

                            # 递归复制嵌套块
                            for block_entity in block_def:
                                if block_entity.dxftype() == 'INSERT':
                                    nested_block_name = block_entity.dxf.name
                                    if nested_block_name not in copied_block_names and nested_block_name in doc.blocks:
                                        nested_block = doc.blocks[nested_block_name]
                                        new_nested_block = new_doc.blocks.new(name=nested_block_name)
                                        for nested_entity in nested_block:
                                            new_nested_block.add_entity(nested_entity.copy())
                                        copied_block_names.add(nested_block_name)

                # 复制实体到新文档
                new_msp.add_entity(entity.copy())

        # 生成输出文件名
        output_path = output_path_template.format(index + 1)

        # 保存新文件
        new_doc.saveas(output_path)

def process_dxf(input_path, output_template):
    """主处理函数"""
    # 获取所有TABLET块的边界和引用数据
    tablet_data = get_tablet_bounds_and_refs(input_path)

    # 读取原始文件
    doc = ezdxf.readfile(input_path)

    # 为每个TABLET区域提取实体并创建新文件
    extract_entities_for_tablet(doc, tablet_data, output_template)

    return len(tablet_data)  # 返回处理的TABLET数量

# 使用示例
if __name__ == "__main__":
    input_path = r"C:\srtp\dxf_split\8-040116-SOP1(ETCH) REV A.dxf"
    output_template = r"C:\srtp\dxf_split\8-040116-SOP1(ETCH) REV A_{}.dxf"

    try:
        print(f"\n开始处理文件: {input_path}")
        if not os.path.exists(input_path):
            print("错误：输入文件不存在！")
        else:
            print("文件存在，开始处理...")
            num_tablets = process_dxf(input_path, output_template)
            print(f"\n已处理 {num_tablets} 个TABLET块，并生成相应的DXF文件。")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()