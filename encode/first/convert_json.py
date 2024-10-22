import json

# Load the JSON data from the file
with open(r'C:\Users\15653\dwg-cx\dataset\modified\split_by_own_remove_space.json', 'r') as file:
    data = json.load(file)

# Define the path for the output TXT file
output_txt_path =r'C:\Users\15653\dwg-cx\dataset\modified\split_by_own_remove_space.txt'

# Write the formatted data to the TXT file
with open(output_txt_path, 'w') as txt_file:
    for entry in data:
        src = entry['src']
        n_num = entry['n_num']
        succs = entry['succs']
        features = entry['features']
        fname = entry['fname']

        # Format the data into one line
        line = f'{{"src": "{src}", "n_num": {n_num}, "succs": {succs}, "features": {features}, "fname": "{fname}"}}\n'

        # Write each entry as one line in the TXT file
        txt_file.write(line)


