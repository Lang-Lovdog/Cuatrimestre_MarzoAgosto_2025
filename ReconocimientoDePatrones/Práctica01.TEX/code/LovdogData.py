import cv2                                  #type: ignore
import numpy                as     np       #type: ignore
import matplotlib.pyplot    as     plt      #type: ignore
from   pathlib              import Path     #type: ignore
from   skimage              import io       #type: ignore
from   skimage              import color    #type: ignore

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

def load_file_with_extra_2dRow(file_path, extra_info=None):
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

def load_file_with_extra_2d(file_path, extra_info=None):
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
        frame_2d = np.loadtxt(file_path, delimiter=delimiter).reshape((rows, cols))
        frames_per_file.append(frame_2d)
                
        print(f"DEBUG: Loaded {len(frames_per_file)} frames from {file_path}")
        return frames_per_file

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def load_image_file(file_path, extra_info=None):
    """
    Loads an image file (PNG, JPG, etc.) and converts it to grayscale numpy array.
    
    Args:
        file_path (str or Path): Path to the image file.
        extra_info (dict, optional): Additional loading instructions.
            May contain:
            - 'resize' (tuple): Target size (width, height) for resizing
            - 'normalize' (bool): Whether to normalize to [0, 1] range
            
    Returns:
        list: A list containing one numpy array (the image)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: Image file not found: {file_path}")
        return []
    
    try:
        # Read image
        image = io.imread(str(file_path))
        
        # Convert to grayscale if it's color
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        
        # Apply transformations based on extra_info
        if extra_info:
            # Resize if specified
            if 'resize' in extra_info:
                target_size = extra_info['resize']
                image = cv2.resize(image, target_size)
            
            # Normalize if specified
            if extra_info.get('normalize', False):
                image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
        
        return [image]
        
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return []

def load_file_with_extra(file_path, extra_info=None):
    """
    Universal file loader that handles both text data and images.
    """
    if extra_info is None:
        raise ValueError("extra_info must be provided")
    
    file_path = Path(file_path)
    
    # Determine file type based on extension or extra_info
    if 'type' in extra_info:
        file_type = extra_info['type']
    else:
        # Auto-detect from file extension
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            file_type = 'image'
        else:
            file_type = '2d'  # Default to text data
    
    # Dispatch to appropriate loader
    if file_type.lower() == 'image':
        return load_image_file(file_path, extra_info)
    elif file_type.lower() == '2d':
        return load_file_with_extra_2d(file_path, extra_info)
    elif file_type.lower() == '2drow':
        return load_file_with_extra_2dRow(file_path, extra_info)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def plot_image_grid(images, titles=None, figsize=(15, 10)):
    """
    Plot a grid of images for visualization.
    
    Args:
        images (list): List of numpy arrays (images)
        titles (list): List of titles for each image
        figsize (tuple): Figure size
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if n_images > 1 else [axes]
    
    for i, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image, cmap='gray')
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_sample_images(data_porter, n_samples=4):
    """
    Visualize sample images from each class.
    """
    images = []
    titles = []
    
    for class_name, frames in data_porter.frames.items():
        for i in range(min(n_samples, len(frames))):
            images.append(frames[i])
            titles.append(f"{class_name} - Sample {i+1}")
    
    plot_image_grid(images, titles)

def translate_px_aug(px, extra_info):
    if extra_info is None:
        extra_info = {}

    offset_x = extra_info.get('offset_x', 1)
    offset_y = extra_info.get('offset_y', 1)
    tremble  = extra_info.get('tremble', True)
    if offset_x == 0 and offset_y == 0:
        return np.array(px)

    px_new = px.copy()  # Start with a copy of the original list

    for p in px:
        # Translate the pixels positively
        px_mod = np.roll(p, shift=offset_y, axis=0)
        px_mod = np.roll(px_mod, shift=offset_x, axis=1)
        np.append(px_new, px_mod)

    if tremble:
        for p in px:
            # Translate the pixels negatively
            px_mod = np.roll(p, shift=-offset_y, axis=0)
            px_mod = np.roll(px_mod, shift=-offset_x, axis=1)
            np.append(px_new, px_mod)

    return np.array(px_new)
