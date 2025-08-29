import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def plot_frame_slides(image, sleep=0.1):
    fig, ax = plt.subplots()
    for frame in image:
        ax.clear()
        ax.imshow(frame, cmap='gray')
        plt.pause(sleep)
    plt.close()

def load_file_with_extra(file_path, extra_info=None):
    """
    Loads a file where each line is a flattened 1D array representing a single frame.
    Reshapes each line into a 2D array based on 'cols' and 'rows' from extra_info.

    Args:
        file_path (str or Path): Path to the file to load.
        extra_info (dict, optional): A dictionary containing instructions for loading.
            Must contain:
            - 'cols' (int): Number of columns for the 2D frame.
            - 'rows' (int): Number of rows for the 2D frame.
            May contain:
            - 'delimiter' (str): Delimiter for text files (e.g., ',', ' '). Defaults to any whitespace.

    Returns:
        list of np.ndarray: A list where each element is a 2D frame (rows, cols) from a line in the file.
                            Returns an empty list if loading fails.
    """
    if extra_info is None:
        extra_info = {}
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return []

    # Get the essential shape information
    try:
        cols = int(extra_info['cols'])
        rows = int(extra_info['rows'])
    except KeyError:
        print(f"Error: 'cols' and 'rows' must be provided in extra_info for file {file_path}.")
        return []
    
    frames_per_file = []
    delimiter = extra_info.get('delimiter', None)
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Convert the line of text into a list of numbers
                data_1d = np.loadtxt([line], delimiter=delimiter)
                
                # Check if the line has the correct number of elements
                expected_elements = cols * rows
                if data_1d.size != expected_elements:
                    print(f"Warning: Line {line_num+1} in {file_path} has {data_1d.size} elements, "
                          f"but {expected_elements} are needed for a {rows}x{cols} frame. Skipping line.")
                    continue
                
                # Reshape the 1D array into a 2D frame
                frame_2d = data_1d.reshape((rows, cols))
                frames_per_file.append(frame_2d)
                
        print(f"DEBUG: Loaded {len(frames_per_file)} frames from {file_path}")
        return frames_per_file

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []
