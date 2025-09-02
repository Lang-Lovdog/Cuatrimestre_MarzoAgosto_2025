import json
from pathlib import Path

# Define the root data directory and subjects for Experiment II
data_root = Path("data/experiment-ii")
subjects = [f"S{i}" for i in range(1, 9)]  # Creates ['S1', 'S2', ..., 'S8']
mat_types = ['Sponge_Mat', 'Air_Mat']

# Define the complex mapping from filename parts to categories
def get_category_from_filename(mat_type, letter, number):
    """
    Determines the class (Supine, Right, Left) based on the filename parts.
    """
    # Convert letter to uppercase for consistency
    letter = letter.upper()
    
    # Define the rules
    if letter == 'B':
        return "Supine"
    elif letter == 'C':
        return "Right"
    elif letter == 'D':
        return "Right"
    elif letter == 'F':
        return "Supine"
    elif letter == 'E':
        # Specific rules for E series
        if number in [1, 2, 5]:
            return "Right"
        elif number in [3, 4, 6]:
            return "Left"
        else:
            return None  # E7, E8, E9, E10 don't have defined categories
    else:
        return None  # Unknown letter

# Initialize the JSON structure
config_dict = {
    "csv": "Features_Experiment-II.csv"
}

# Initialize category dictionaries
categories = {
    "Supine": {"class": "Supine", "path": [], "extra": {"cols": 27, "rows": 64}},
    "Right": {"class": "Right", "path": [], "extra": {"cols": 27, "rows": 64}},
    "Left": {"class": "Left", "path": [], "extra": {"cols": 27, "rows": 64}}
}

# Scan through all possible files and categorize them
for subject in subjects:
    for mat_type in mat_types:
        subject_mat_dir = data_root / subject / mat_type
        
        if not subject_mat_dir.exists():
            print(f"Warning: Directory {subject_mat_dir} not found. Skipping.")
            continue
            
        # Look for files in the format "Matrix_T_LX"
        for file_path in subject_mat_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith("Matrix_") and file_path.suffix == '.txt':
                # Parse the filename to extract T, L, and X
                parts = file_path.stem.split('_')  # e.g., ['Matrix', 'Air', 'B1']
                
                if len(parts) >= 3:
                    file_mat_type = parts[1]  # 'Air' or 'Sponge'
                    letter_number = parts[2]  # 'B1', 'C2', etc.
                    
                    # Extract letter and number
                    if letter_number and len(letter_number) >= 2:
                        letter = letter_number[0]  # 'B', 'C', etc.
                        number_str = letter_number[1:]  # '1', '2', etc.
                        
                        try:
                            number = int(number_str)
                            
                            # Get the category for this file
                            category = get_category_from_filename(file_mat_type, letter, number)
                            
                            if category:
                                # Add the file path to the appropriate category
                                categories[category]["path"].append(str(file_path))
                            else:
                                print(f"Warning: No category defined for file {file_path}. Skipping.")
                                
                        except ValueError:
                            print(f"Warning: Cannot parse number from {letter_number} in file {file_path}. Skipping.")
                else:
                    print(f"Warning: Unexpected filename format: {file_path.name}. Skipping.")

# Add the categories to the main config dictionary
for category_name, category_data in categories.items():
    config_dict[category_name] = category_data

# Save the configuration to a JSON file
output_json_path = "config_experiment_ii.json"
with open(output_json_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"Successfully generated JSON config: {output_json_path}")
print("File counts per category:")
for category_name, category_data in categories.items():
    print(f"  {category_name}: {len(category_data['path'])} files")
