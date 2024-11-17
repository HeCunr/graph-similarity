import os

def count_dxf_files(directory):
    dxf_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                dxf_count += 1
    return dxf_count

# 使用方法：将 "your_directory_path" 替换为目标目录路径
directory_path = r"/mnt/share/DeepDXF_CGMN/encode/data/DeepDXF/dxf_vec_4096"
dxf_file_count = count_dxf_files(directory_path)
print(f"目录 '{directory_path}' 中的 .dxf 文件个数为: {dxf_file_count}")
