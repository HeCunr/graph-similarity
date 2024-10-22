import ezdxf
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def get_normalized_entity_handles(dxf_file_path):
    doc = ezdxf.readfile(dxf_file_path)
    msp = doc.modelspace()
    entity_handles = []
    count = 0
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type in ['ARC', 'TEXT', 'MTEXT', 'LWPOLYLINE', 'INSERT', 'DIMENSION', 'LEADER', 'CIRCLE', 'HATCH', 'LINE']:
            print(entity_type, entity.dxf.handle)
            count += 1
            handle = int(entity.dxf.handle, 16)  # 将十六进制转换为十进制
            print(handle)
            entity_handles.append((entity_type, handle))
    print(f"共找到{count}个实体")
    # 按句柄值排序
    entity_handles.sort(key=lambda x: x[1])

    # 预处理句柄值，使其连续
    processed_handles = []
    base_handle = entity_handles[0][1]  # 获取第一个句柄值作为基准

    for i, (entity_type, handle) in enumerate(entity_handles):
        new_handle = base_handle + i  # 生成连续的新句柄值
        processed_handles.append((entity_type, new_handle))
        # print(f"Original handle: {hex(handle)} -> New handle: {hex(new_handle)}")  # 打印转换信息

    # 使用Min-Max归一化方法将处理后的句柄值归一化到0-1范围
    handles = [h for _, h in processed_handles]
    min_handle = min(handles)
    max_handle = max(handles)
    normalized_handles = [(t, (h - min_handle) / (max_handle - min_handle)) for t, h in processed_handles]

    return normalized_handles, processed_handles

def create_adjacency_matrix(entity_handles):
    entity_types = sorted(set(t for t, _ in entity_handles))
    type_to_index = {t: i for i, t in enumerate(entity_types)}
    n = len(entity_types)

    adjacency_matrix = np.zeros((n, n), dtype=int)
    edge_weights = np.zeros((n, n), dtype=int)

    for i in range(len(entity_handles) - 1):
        current_type, _ = entity_handles[i]
        next_type, _ = entity_handles[i + 1]
        current_index = type_to_index[current_type]
        next_index = type_to_index[next_type]

        adjacency_matrix[current_index, next_index] = 1
        adjacency_matrix[next_index, current_index] = 1
        edge_weights[current_index, next_index] += 1
        edge_weights[next_index, current_index] += 1

    return adjacency_matrix, edge_weights, entity_types

import matplotlib.pyplot as plt

def visualize_entities_by_handle(normalized_handles):
    # 将数据分离为两个列表
    entity_types, handle_values = zip(*normalized_handles)

    # 创建颜色映射
    unique_types = list(set(entity_types))
    color_map = plt.cm.get_cmap('tab20')
    colors = [color_map(i/len(unique_types)) for i in range(len(unique_types))]
    type_to_color = dict(zip(unique_types, colors))

    # 创建图形，增加高度
    fig, ax = plt.subplots(figsize=(12, 20))

    # 绘制水平条形图
    bars = ax.barh(range(len(handle_values)), handle_values, align='center',
                   color=[type_to_color[t] for t in entity_types])

    # 设置y轴标签
    ax.set_yticks(range(len(entity_types)))
    ax.set_yticklabels([f"{t}: {v:.4f}" for t, v in normalized_handles], fontsize=5)

    # 旋转y轴标签并调整位置
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", va="center")
    ax.tick_params(axis='y', which='major', pad=40)

    # 设置x轴范围和标签
    ax.set_xlim(0, 1)
    ax.set_xlabel("归一化句柄值")

    # 设置标题
    ax.set_title("实体按归一化句柄值排序")

    # 添加图例
    handles = [plt.Rectangle((0,0),1,1, color=type_to_color[t]) for t in unique_types]
    ax.legend(handles, unique_types, loc='lower right', title="实体类型", bbox_to_anchor=(1.1, 0))

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("entities_by_handle.png", dpi=300, bbox_inches='tight')
    plt.close()
def visualize_entities_by_type(type_dict):
    plt.figure(figsize=(12, 6))
    plt.title("实体按类型分组")
    for i, (entity_type, handles) in enumerate(type_dict.items()):
        plt.scatter([h for h in handles], [i] * len(handles), label=entity_type)
        plt.text(-0.1, i, entity_type, ha='right', va='center')
    plt.xlim(0, 1)
    plt.xlabel("归一化句柄值")
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig("entities_by_type.png")
    plt.close()

def visualize_adjacency_matrix(adjacency_matrix, entity_types):
    plt.figure(figsize=(10, 8))
    plt.imshow(adjacency_matrix, cmap='Blues')
    plt.title("邻接矩阵")
    plt.xticks(range(len(entity_types)), entity_types, rotation=90)
    plt.yticks(range(len(entity_types)), entity_types)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("adjacency_matrix.png")
    plt.close()

def visualize_edge_weights(edge_weights, entity_types):
    plt.figure(figsize=(10, 8))
    plt.imshow(edge_weights, cmap='YlOrRd')
    plt.title("边权矩阵")
    plt.xticks(range(len(entity_types)), entity_types, rotation=90)
    plt.yticks(range(len(entity_types)), entity_types)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("edge_weights.png")
    plt.close()

def visualize_node_edges(adjacency_matrix, entity_types):
    G = nx.Graph(adjacency_matrix)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')

    # 添加节点标签
    labels = {i: f"{entity_types[i]}\n边数: {sum(adjacency_matrix[i])}" for i in range(len(entity_types))}
    nx.draw_networkx_labels(G, pos, labels)

    plt.title("实体关系图")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("node_edges.png")
    plt.close()

def main(dxf_file_path):
    normalized_handles, original_handles = get_normalized_entity_handles(dxf_file_path)
    adjacency_matrix, edge_weights, entity_types = create_adjacency_matrix(original_handles)

    # 输出1：按句柄值排序的实体列表
    print("按句柄值排序的实体列表:")
    print([[entity_type, f"{handle:.4f}"] for entity_type, handle in normalized_handles])
    visualize_entities_by_handle(normalized_handles)

    # 输出2：按实体类型分组的句柄值列表
    type_dict = defaultdict(list)
    for entity_type, handle in normalized_handles:
        type_dict[entity_type].append(handle)
    print("\n按实体类型分组的句柄值列表:")
    print([[entity_type, [f"{h:.4f}" for h in handles]] for entity_type, handles in type_dict.items()])
    visualize_entities_by_type(type_dict)

    # 输出3：邻接矩阵
    print("\n邻接矩阵:")
    print(adjacency_matrix.tolist())
    visualize_adjacency_matrix(adjacency_matrix, entity_types)

    # 输出4：边权矩阵
    print("\n边权矩阵:")
    print(edge_weights.tolist())
    visualize_edge_weights(edge_weights, entity_types)

    # 输出5：节点边数
    print("\n节点边数:")
    node_edges = [[entity_type, sum(adjacency_matrix[i])] for i, entity_type in enumerate(entity_types)]
    print(node_edges)
    visualize_node_edges(adjacency_matrix, entity_types)

if __name__ == '__main__':
    dxf_file_path = r'C:\Users\15653\dwg-cx\dataset\modified\DFN6TLCT(NiPdAu)（321） -551 Rev1_2.dxf'  # 请替换为您的DXF文件路径
    main(dxf_file_path)