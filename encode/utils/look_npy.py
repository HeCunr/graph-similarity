# !/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Function to load and view the content of a npy file
def view_npy_file(file_path):
    try:
        # Load the npy file
        data = np.load(file_path)
        print("Content of the npy file:\n", data)
    except Exception as e:
        print(f"Error loading file: {str(e)}")

# Example usage
npy_file_path ='/mnt/share/CGMN/CGMN/data/CFG/OpenSSL_11ACFG_min3_max13/acfgSSL_6/class_perm.npy'  # Replace with your file path
view_npy_file(npy_file_path)
