
import h5py
import numpy as np

def view_h5_contents(file_path):
    # 设置numpy打印选项，禁用省略号
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("\nDatasets:")

        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  - {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")

                # 处理不同维度的数据集
                if len(obj.shape) == 1:
                    print(f"    All elements:")
                    for i, value in enumerate(obj[:]):
                        print(f"      [{i}]: {value}")
                elif len(obj.shape) == 2:
                    print(f"    All rows:")
                    for i, row in enumerate(obj[:]):
                        print(f"      [{i}]: {row}")
                elif len(obj.shape) == 3:
                    print(f"    All elements (3D):")
                    for i in range(obj.shape[0]):
                        print(f"    Sample {i}:")
                        for j in range(obj.shape[1]):
                            print(f"      Row {j}: {obj[i][j]}")
                print()

        f.visititems(print_dataset_info)

        # 打印文件属性（如果有）
        if f.attrs:
            print("\nFile Attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    # 根据需要修改文件路径
    file_path = r'C:\srtp\encode\data\DeepDXF\dxf_vec\DFN6BU(NiPdAu)-437 Rev1_2.h5'
    view_h5_contents(file_path)