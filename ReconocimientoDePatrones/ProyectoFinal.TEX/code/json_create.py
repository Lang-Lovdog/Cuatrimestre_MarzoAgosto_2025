import json
from pathlib import Path

# Define the mapping from file number to your 3 categories
posture_mapping = {
    "Supine": [1, 8, 9, 10, 11, 12, 15, 16, 17],
    "Right": [2, 4, 5, 13],
    "Left": [3, 6, 7, 14]
}

# Define the root data directory and subjects
data_root = Path("data/experiment-i")
subjects = [f"S{i}" for i in range(1, 14)] # Creates ['S1', 'S2', ..., 'S13']

# Initialize the JSON structure
config_dict = {
    "csv": "Features_Experiment-I.csv"
}

# For each category, find all matching files across all subjects
for category, numbers in posture_mapping.items():
    path_list = []
    
    for subject in subjects:
        subject_dir = data_root / subject
        if not subject_dir.exists():
            print(f"Warning: Subject directory {subject_dir} not found. Skipping.")
            continue
            
        # For each file number in this category, check if the file exists
        for num in numbers:
            file_name = f"{num}.txt"
            file_path = subject_dir / file_name
            
            if file_path.exists():
                # Add the path as a string, using forward slashes for compatibility
                path_list.append(str(file_path))
            else:
                print(f"Warning: File {file_path} not found. Skipping.")
    
    # Add the category to the main config dictionary
    config_dict[category] = {
        "class": category,
        "path": path_list,
        "extra": {
            "cols": 32,
            "rows": 64
        }
    }

# Save the configuration to a JSON file
output_json_path = "config_experiment_i.json"
with open(output_json_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"Successfully generated JSON config: {output_json_path}")
