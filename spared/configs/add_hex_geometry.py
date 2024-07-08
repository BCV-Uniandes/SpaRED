import json
import os

# Specify the directory containing your JSON files
directory_path = '/home/dvegaa/spared/spared/configs'

# List all JSON files in the directory
json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

# Iterate through each file
for filename in json_files:
    file_path = os.path.join(directory_path, filename)

    # Open and load the JSON data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Add the new key-value pair
    data["hex_geometry"] = True

    # Write the modified data back to the same file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

print("All files have been updated.")