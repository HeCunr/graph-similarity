#utils/deepdxf_augment.py
import os
import h5py
import numpy as np

############################################
# 1. 数据增强的核心：随机扰动、平移、缩放等
############################################

def clip_value(v):
    """用于在 [0, 255] 范围内裁剪。"""
    return max(0, min(255, v))

def random_shift_int(original_value, shift_range=5):
    """对于 [0,255] 区间内的整数做随机平移，加减不超过 shift_range。"""
    if original_value < 0:
        # -1 表示无效特征，不做任何修改
        return original_value
    delta = np.random.randint(-shift_range, shift_range + 1)
    return clip_value(original_value + delta)

def random_scale_int(original_value, scale_min=0.9, scale_max=1.1):
    """对于 [0,255] 区间内的整数做随机缩放。"""
    if original_value < 0:
        return original_value
    factor = np.random.uniform(scale_min, scale_max)
    new_val = int(round(original_value * factor))
    return clip_value(new_val)

def random_toggle_01(original_value, prob=0.1):
    """对于 0/1 的特征，小概率翻转; 其他值不处理。"""
    if original_value not in [0,1]:
        return original_value
    if np.random.rand() < prob:
        return 1 - original_value  # 0 <-> 1
    return original_value

############################################
# 2. ENTITY_TYPES & FEATURE 下标定义
############################################

ENTITY_TYPES = [
    'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT',
    'MTEXT', 'HATCH', 'DIMENSION', 'LEADER', 'INSERT',
    'SPLINE', 'SOLID', 'EOS'
]
# 请注意：代码中第0列存储 entity_type_idx，范围0~11，或12表示EOS。
# 若取值为 -1，表示此行无效/填充，通常不应出现 -1 作为实体类型，但在有些场景下可能存储了-1。后续将-1替换为12。

# FEATURE_NAMES 的顺序（下标从 0 开始，一共有 43 个）：
FEATURE_NAMES = [
    'solid_fill',         # 1
    'associative',        # 2
    'boundary_paths',     # 3

    'text_insert_point_x',# 4
    'text_insert_point_y',# 5
    'height',             # 6
    'text_rotation',      # 7

    'mtext_insert_point_x',#8
    'mtext_insert_point_y',#9
    'char_height',        #10
    'width',              #11

    'closed',             #12
    'points_x',           #13
    'points_y',           #14
    'count',              #15

    'arc_center_x',       #16
    'arc_center_y',       #17
    'arc_radius',         #18
    'start_angle',        #19
    'end_angle',          #20

    'start_point_x',      #21
    'start_point_y',      #22
    'end_point_x',        #23
    'end_point_y',        #24

    'circle_center_x',    #25
    'circle_center_y',    #26
    'circle_radius',      #27

    'defpoint_x',         #28
    'defpoint_y',         #29
    'text_midpoint_x',    #30
    'text_midpoint_y',    #31

    'vertices_x',         #32
    'vertices_y',         #33

    'insert_insert_point_x', #34
    'insert_insert_point_y', #35
    'scale_x',            #36
    'scale_y',            #37
    'insert_rotation',    #38

    'control_points_x',   #39
    'control_points_y',   #40
    'avg_knots',          #41

    'solid_points_x',     #42
    'solid_points_y'      #43
]

# 便于查找列号用的 dict（但此处通常直接写死列号也行）
feature_index_map = {name: i for i, name in enumerate(FEATURE_NAMES)}

############################################
# 3. 单行实体的增强逻辑
############################################

def augment_entity_line(entity_row):
    """
    针对单行实体（形状为 (44,) 的 numpy 数组）。
    - 第0列: entity_type_idx (int)
    - 第1~43列: 对应 FEATURE_NAMES
    """

    # 强制把实体类型转换为 int，避免浮点索引
    entity_row[0] = int(entity_row[0])
    entity_type_idx = entity_row[0]

    # 若类型无效/越界 或是 EOS(12)，则不增强
    if entity_type_idx < 0 or entity_type_idx >= len(ENTITY_TYPES):
        return entity_row
    if entity_type_idx == 12:
        return entity_row

    entity_type = ENTITY_TYPES[entity_type_idx]

    # ================ LINE ================
    # LINE: start_point_x=21, start_point_y=22, end_point_x=23, end_point_y=24
    if entity_type == 'LINE':
        for feat_idx in [21, 22, 23, 24]:
            col = feat_idx  # <-- 不再 +1
            new_val = random_shift_int(entity_row[col], shift_range=5)
            if new_val != -1:
                new_val = int(new_val)
            entity_row[col] = new_val

    # ================ CIRCLE ================
    elif entity_type == 'CIRCLE':
        # center_x=25, center_y=26, radius=27
        for feat_idx in [25, 26]:
            col = feat_idx
            new_val = random_shift_int(entity_row[col], shift_range=5)
            if new_val != -1:
                new_val = int(new_val)
            entity_row[col] = new_val

        col_radius = 27  # => 27
        if entity_row[col_radius] >= 0:
            if np.random.rand() < 0.5:
                r = random_shift_int(entity_row[col_radius], shift_range=5)
            else:
                r = random_scale_int(entity_row[col_radius], 0.9, 1.1)
            entity_row[col_radius] = int(r) if r != -1 else -1

    # ================ ARC ================
    elif entity_type == 'ARC':
        # arc_center_x=16, arc_center_y=17, arc_radius=18, start_angle=19, end_angle=20
        for feat_idx in [16, 17]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

        col_radius = 18
        if entity_row[col_radius] >= 0:
            if np.random.rand() < 0.5:
                rr = random_shift_int(entity_row[col_radius], shift_range=5)
            else:
                rr = random_scale_int(entity_row[col_radius], 0.9, 1.1)
            entity_row[col_radius] = int(rr) if rr != -1 else -1

        for feat_idx in [19, 20]:
            col = feat_idx
            ang = random_shift_int(entity_row[col], shift_range=10)
            entity_row[col] = int(ang) if ang != -1 else -1

    # ================ LWPOLYLINE ================
    elif entity_type == 'LWPOLYLINE':
        # closed=11, points_x=12, points_y=13, count=14
        col_closed = 11
        cv = random_toggle_01(entity_row[col_closed], prob=0.1)
        entity_row[col_closed] = int(cv) if cv != -1 else -1

        for feat_idx in [12, 13]:
            col = feat_idx
            pv = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(pv) if pv != -1 else -1

        col_count = 14
        if entity_row[col_count] >= 0:
            delta = np.random.randint(-2, 3)
            new_val = clip_value(entity_row[col_count] + delta)
            entity_row[col_count] = int(new_val)

    # ================ TEXT ================
    elif entity_type == 'TEXT':
        # text_insert_point_x=3, text_insert_point_y=4, height=5, text_rotation=6
        for feat_idx in [3, 4]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

        col_height = 5
        valh = random_shift_int(entity_row[col_height], shift_range=5)
        entity_row[col_height] = int(valh) if valh != -1 else -1

        col_rot = 6
        rotv = random_shift_int(entity_row[col_rot], shift_range=10)
        entity_row[col_rot] = int(rotv) if rotv != -1 else -1

    # ================ MTEXT ================
    elif entity_type == 'MTEXT':
        for feat_idx in [7, 8]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

        for feat_idx in [9, 10]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

    # ================ HATCH ================
    elif entity_type == 'HATCH':
        # solid_fill=0, associative=1, boundary_paths=2
        for feat_idx in [0, 1]:
            col = feat_idx
            val = random_toggle_01(entity_row[col], prob=0.1)
            entity_row[col] = int(val) if val != -1 else -1

        col_bp = 2
        if entity_row[col_bp] >= 0:
            delta = np.random.randint(-2, 3)
            new_val = clip_value(entity_row[col_bp] + delta)
            entity_row[col_bp] = int(new_val)

    # ================ DIMENSION ================
    elif entity_type == 'DIMENSION':
        # defpoint_x=28, defpoint_y=29, text_midpoint_x=30, text_midpoint_y=31
        for feat_idx in [28, 29, 30, 31]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

    # ================ LEADER ================
    elif entity_type == 'LEADER':
        for feat_idx in [32, 33]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

    # ================ INSERT ================
    elif entity_type == 'INSERT':
        for feat_idx in [34, 35]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

        for feat_idx in [36, 37]:  # scale_x, scale_y
            col = feat_idx
            if entity_row[col] >= 0:
                if np.random.rand() < 0.5:
                    new_val = random_shift_int(entity_row[col], shift_range=10)
                else:
                    new_val = random_scale_int(entity_row[col], 0.9, 1.1)
                entity_row[col] = int(new_val)

        col_rot = 38
        rv = random_shift_int(entity_row[col_rot], shift_range=10)
        entity_row[col_rot] = int(rv) if rv != -1 else -1

    # ================ SPLINE ================
    elif entity_type == 'SPLINE':
        for feat_idx in [39, 40]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

        col_knots = 41
        if entity_row[col_knots] >= 0:
            if np.random.rand() < 0.5:
                kv = random_shift_int(entity_row[col_knots], shift_range=10)
            else:
                kv = random_scale_int(entity_row[col_knots], 0.9, 1.1)
            entity_row[col_knots] = int(kv)

    # ================ SOLID ================
    elif entity_type == 'SOLID':
        # solid_points_x=42, solid_points_y=43
        for feat_idx in [42, 43]:
            col = feat_idx
            val = random_shift_int(entity_row[col], shift_range=5)
            entity_row[col] = int(val) if val != -1 else -1

    return entity_row



############################################
# 4. 序列级别的操作：shuffle、delete、duplicate
############################################

def shuffle_entities(entities):
    """
    轻度打乱：随机交换相邻行，次数约为有效行数的一半。
    entities: (N,44) 的数组，N为实体行数(不含EOS)。
    """
    n = len(entities)
    num_swaps = n // 2
    for _ in range(num_swaps):
        i = np.random.randint(0, n-1)
        entities[i], entities[i+1] = entities[i+1].copy(), entities[i].copy()
    return entities

def random_delete(entities, delete_ratio=0.05):
    """
    随机删除一定比例的实体行。返回新的 (M,44) 数组。
    """
    n = len(entities)
    keep_mask = np.random.rand(n) > delete_ratio
    return entities[keep_mask]

def random_duplicate(entities, duplicate_ratio=0.05):
    """
    随机复制一定比例的行。复制时对复制出来的行再做一次 augment_entity_line()。
    """
    n = len(entities)
    num_dup = int(n * duplicate_ratio)
    if num_dup <= 0:
        return entities

    chosen_indices = np.random.choice(np.arange(n), size=num_dup, replace=True)
    duplicates = []
    for idx in chosen_indices:
        new_entity = entities[idx].copy()
        # 再次扰动复制出来的实体
        new_entity = augment_entity_line(new_entity)
        duplicates.append(new_entity)

    augmented = np.concatenate([entities, np.array(duplicates)], axis=0)
    return augmented

def fix_length_and_append_eos(entities, max_len=4096):
    """
    将 entities (K,44) 填充或截断到 (max_len,44)，并在末尾加一行 EOS。
    若实际K>=max_len，则只保留前 (max_len-1) 行 + 1 行EOS。
    若K<max_len，则中间用 -1 填充，直到最后也补上EOS。

    最后，将所有无效行的第0列（实体类型）设置为 12(EOS)，
    但其余参数列依旧保持 -1。
    """
    out = np.full((max_len, 44), -1, dtype=np.int16)
    eos_idx = ENTITY_TYPES.index('EOS')
    eos_line = np.full((44,), -1, dtype=np.int16)
    eos_line[0] = eos_idx

    if len(entities) >= max_len:
        # 截断输入数据到前 4095 行，并确保列数为 44
        out[:max_len-1, :] = entities[:max_len-1, :44]
        out[max_len-1] = eos_line
    else:
        # 确保输入数据的列数为 44
        out[:len(entities), :44] = entities[:, :44]
        # 在下一行写入 EOS
        if len(entities) < max_len:
            out[len(entities)] = eos_line

    # ---- 在此处将第一列 <0 的地方统一置为 12(EOS) ----
    #     这样能保证无效行的“实体类型”列始终为 12
    invalid_mask = (out[:, 0] < 0)
    out[invalid_mask, 0] = eos_idx

    return out

############################################
# 5. augment_sample: 对单个样本(4096,44)做增强
############################################

def augment_sample(dxf_arr, do_shuffle=True, delete_ratio=0.05, duplicate_ratio=0.05):
    """
    对单个样本 dxf_arr (shape=(4096,44)) 做数据增强。
    返回相同形状的 numpy 数组 (4096,44)。
    """
    # 1. 找到前 N 行有效实体（直到遇到 entity_type=12 或到达末尾）
    max_seq_len = dxf_arr.shape[0]  # 4096
    i = 0
    while i < max_seq_len and dxf_arr[i, 0] != 12:
        i += 1
    # i 即有效实体的数量
    valid_entities = dxf_arr[:i].copy()  # shape=(N,44)

    # 2. 对每个实体行做列级别增强
    for row_idx in range(len(valid_entities)):
        valid_entities[row_idx] = augment_entity_line(valid_entities[row_idx])

    # 3. 序列级别可选操作
    if do_shuffle and len(valid_entities) > 1:
        valid_entities = shuffle_entities(valid_entities)
    if delete_ratio > 0:
        valid_entities = random_delete(valid_entities, delete_ratio)
    if duplicate_ratio > 0:
        valid_entities = random_duplicate(valid_entities, duplicate_ratio)

    # 4. 最后补齐或截断到 4096, 并加 EOS 行
    out = fix_length_and_append_eos(valid_entities, max_len=4096)
    assert out.shape == (4096, 44), f"输出形状错误: {out.shape}"
    return out.astype(np.int32)  # 转换为 int32 以兼容后续处理

############################################
# 6. 对整个 H5 文件处理
############################################

def augment_h5_dataset(input_h5_path, output_h5_path,
                       shuffle=True, delete_ratio=0.05, duplicate_ratio=0.05):
    """
    对单个 H5 文件进行数据增强。
    其中 'dxf_vec' 数据集形状为 (num_samples, 4096, 44)。
    增强后写入新的 output_h5_path。
    """
    with h5py.File(input_h5_path, 'r') as fin, \
            h5py.File(output_h5_path, 'w') as fout:

        if 'dxf_vec' not in fin:
            print(f"[Warning] {input_h5_path} 中没有 'dxf_vec' 数据集，跳过。")
            return

        dset_in = fin['dxf_vec']
        num_samples = len(dset_in)

        # 创建输出数据集
        dset_out = fout.create_dataset(
            'dxf_vec',
            shape=(num_samples, 4096, 44),
            dtype=np.int16
        )

        for i in range(num_samples):
            original_sample = dset_in[i]  # shape=(4096,44)
            aug_sample = augment_sample(
                original_sample,
                do_shuffle=shuffle,
                delete_ratio=delete_ratio,
                duplicate_ratio=duplicate_ratio
            )
            dset_out[i] = aug_sample

            if i % 50 == 0:
                print(f"  -> {input_h5_path}: Augmented sample {i}/{num_samples}")

    print(f"完成数据增强: {input_h5_path} -> {output_h5_path}")

############################################
# 7. 批量处理目录中的全部 .h5
############################################

def augment_all_h5_in_directory(input_dir, output_dir,
                                shuffle=True, delete_ratio=0.05, duplicate_ratio=0.05):
    """
    遍历 input_dir 下的所有 .h5 文件，执行数据增强，
    并将结果保存到 output_dir 下相应文件名。
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.h5')]
    if not all_files:
        print(f"在目录 {input_dir} 中未找到任何 .h5 文件。")
        return

    for h5_file in all_files:
        in_path = os.path.join(input_dir, h5_file)
        out_path = os.path.join(output_dir, h5_file)
        print(f"[处理文件]: {in_path}")
        augment_h5_dataset(
            input_h5_path=in_path,
            output_h5_path=out_path,
            shuffle=shuffle,
            delete_ratio=delete_ratio,
            duplicate_ratio=duplicate_ratio
        )

############################################
# 8. 如果需要从命令行运行，可自行添加 argparse
############################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepDXF数据增强脚本")
    parser.add_argument("--input_dir", type=str, default=r"/home/vllm/encode/data/DeepDXF/augu1",help="输入目录，包含原始 .h5 文件")
    parser.add_argument("--output_dir", type=str, default=r"/home/vllm/encode/data/DeepDXF/augu2",help="输出目录，保存增强后的 .h5 文件")
    parser.add_argument("--delete_ratio", type=float, default=0.05, help="随机删除比例")
    parser.add_argument("--duplicate_ratio", type=float, default=0.05, help="随机复制比例")
    parser.add_argument("--disable_shuffle", action='store_true', help="是否禁用随机洗牌")
    args = parser.parse_args()

    do_shuffle = not args.disable_shuffle

    augment_all_h5_in_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        shuffle=do_shuffle,
        delete_ratio=args.delete_ratio,
        duplicate_ratio=args.duplicate_ratio
    )
