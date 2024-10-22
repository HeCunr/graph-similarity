import h5py
import numpy as np

def view_h5_contents(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("\nDatasets:")

        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  - {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")

                # Print first 10 rows or elements
                if len(obj.shape) == 1:
                    print(f"    First 10 elements:")
                    for i, value in enumerate(obj[:10]):
                        print(f"      [{i}]: {value}")
                elif len(obj.shape) == 2:
                    print(f"    First 10 rows:")
                    for i, row in enumerate(obj[:10]):
                        print(f"      [{i}]: {row}")
                else:
                    print(f"    First element: {obj[0]}")
                print()

        f.visititems(print_dataset_info)

        # Print attributes if any
        if f.attrs:
            print("\nFile Attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # file_path = r'C:\srtp\encode\four\part1\dxf_vecs.h5'  # Replace with your H5 file path
    file_path = r'C:\srtp\encode\data\DeepDXF\dxf_vec\DFN6BU(NiPdAu)-437 Rev1_1.h5'
    view_h5_contents(file_path)