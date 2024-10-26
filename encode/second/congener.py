# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from adj_row_col import get_normalized_feature_matrix
import json

# Directory paths
dxf_dir = r'C:\Users\15653\dwg-cx\dataset\modified\valid_new'
output_files = [os.path.join(dxf_dir, f'output_{i}.json') for i in range(1, 7)]  # Changed to .json

# Track files that have already been processed for each output file
processed_files = [set() for _ in range(6)]  # A list of 6 sets

# Function to generate adjacency list
def feature_to_adj(feature_matrix):
    adj_list = []
    for i, row in enumerate(feature_matrix):
        adj_list.append([j for j, val in enumerate(row) if val > 0])
    return adj_list

# Function to drop features
def drop_features(x, drop_prob=0.1):
    drop_mask = torch.rand(x.shape[1]) > drop_prob
    x_dropped = x.clone()
    x_dropped[:, ~drop_mask] = 0
    return x_dropped

# Process each file in directory
for dxf_file in os.listdir(dxf_dir):
    if dxf_file.endswith('.dxf'):
        file_path = os.path.join(dxf_dir, dxf_file)
        feature_matrix = np.array(get_normalized_feature_matrix(file_path))  # Convert to NumPy array

        src_name = dxf_file.replace(' ', '')
        fname = os.path.splitext(src_name)[0]

        # Skip the first column for TEMP matrix
        temp = feature_matrix[:, 1:]

        # Generate six feature matrices and adjacency lists
        for i in range(6):
            if src_name not in processed_files[i]:  # Check if the file is already processed for this output
                # Apply random feature dropping
                temp_dropped = drop_features(torch.tensor(temp, dtype=torch.float32)).numpy()

                # Create the adjacency list
                adj_list = feature_to_adj(temp_dropped)

                # Append the first column and TEMP for the new feature matrix
                f_matrix = np.hstack((feature_matrix[:, :1], temp_dropped))

                # Prepare the data dictionary
                data = {
                    "src": src_name,
                    "n_num": len(feature_matrix),
                    "succs": adj_list,
                    "features": f_matrix.tolist(),
                    "fname": fname
                }

                # Save to respective files
                with open(output_files[i], 'a', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')

                processed_files[i].add(src_name)  # Mark the file as processed for this output file

print("Processing complete. Output files have been saved as JSON.")