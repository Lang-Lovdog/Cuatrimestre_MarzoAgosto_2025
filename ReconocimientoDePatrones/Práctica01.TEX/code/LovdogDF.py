import json
import numpy         as np
import pandas        as pd
import random
import os
from   pathlib       import Path                 #type: ignore
from   collections   import Counter              #type: ignore
from   sklearn.utils import resample             #type: ignore
from   LovdogData    import plot_image           #type: ignore
from   LovdogData    import load_file_with_extra #type: ignore

import json
import numpy as np
import pandas as pd
from   pathlib import Path
from   typing import Dict, List, Optional, Union
from   LovdogData import load_file_with_extra

class LovdogDataFrames:
    def __init__(self, json_file_path=None, loader_function=None):
        self.json_file_path = json_file_path
        self.loader_function = loader_function or load_file_with_extra
        self.json_data = None
        self.feature_table_name = ""
        self.frames = {}
        self.extra_data = {}
        self.class_names = []

        if json_file_path:
            self.load_json(json_file_path)

    def load_json(self, json_file_path):
        """Load and parse JSON configuration file."""
        self.json_file_path = json_file_path
        with open(json_file_path, 'r') as f:
            self.json_data = json.load(f)
        self._parse_json()

    def _parse_json(self):
        """Internal method to parse JSON data."""
        if not self.json_data:
            raise ValueError("No JSON data loaded. Call load_json() first.")

        self.frames = {}
        self.extra_data = {}
        self.class_names = []

        self.feature_table_name = self.json_data.get("csv", "Features_Table.csv")

        for key, value in self.json_data.items():
            if key == "csv":
                continue

            if isinstance(value, dict):
                class_name = value.get("class", f"Unknown_Class_{key}")
                paths = value.get("path", [])
                extra = value.get("extra", {})

                self.extra_data[class_name] = extra
                self.class_names.append(class_name)
                self.frames[class_name] = []

                print(f"[INFO] Loading data for class: {class_name}")
                for path in paths:
                    path_obj = Path(path)
                    if path_obj.is_file():
                        loaded_data = self.loader_function(path_obj, extra)
                        self._add_loaded_data(loaded_data, class_name)
                    elif path_obj.is_dir():
                        for file_path in path_obj.iterdir():
                            if file_path.is_file():
                                loaded_data = self.loader_function(file_path, extra)
                                self._add_loaded_data(loaded_data, class_name)

    def _add_loaded_data(self, loaded_data, class_name):
        """Helper method to handle loader function results."""
        if loaded_data is None:
            return
        if isinstance(loaded_data, list):
            self.frames[class_name].extend(loaded_data)
        else:
            self.frames[class_name].append(loaded_data)

    def get_data_dict(self) -> Dict[str, List[np.ndarray]]:
        """Return the frames dictionary with numpy arrays."""
        return self.frames.copy()

    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        return self.class_names.copy()

    def get_total_samples(self) -> int:
        """Return total number of samples across all classes."""
        return sum(len(frames) for frames in self.frames.values())

    def balance_classes(self, method='undersample', random_state=None):
        """Balance classes using undersampling or oversampling."""
        # Implementation from your original code
        pass

    def get_feature_extractor_input(self) -> Dict[str, List[np.ndarray]]:
        """
        Prepare data for feature extractors.
        Ensures all frames are numpy arrays and properly formatted.
        """
        processed_data = {}
        for class_name, frames in self.frames.items():
            processed_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    processed_frames.append(frame)
                elif hasattr(frame, 'numpy'):  # Handle PyTorch tensors etc.
                    processed_frames.append(frame.numpy())
                else:
                    # Convert other types to numpy arrays
                    processed_frames.append(np.array(frame))
            processed_data[class_name] = processed_frames
        return processed_data

    # Add these methods to the LovdogDataFrames class:

    def augmentation_data(self, augmentation_function, extra_info=None):
        class_counts = {cls: len(frames) for cls, frames in self.frames.items()}
        print(f"Original class distribution: {class_counts}")
        for class_name, frames in self.frames.items():
            self.frames[class_name] = augmentation_function(frames, extra_info)
        new_counts = {cls: len(frames) for cls, frames in self.frames.items()}
        print(f"Augmented class distribution: {new_counts}")

    def balance_frames(self, method='undersample', random_state=None):
        """
        Balance the frames at the data loading level (before feature extraction).
        
        Parameters:
        -----------
        method : str, default='undersample'
            Method to balance classes: 'undersample' or 'oversample'
        
        random_state : int, optional
            Random seed for reproducibility
        """
        if not self.frames:
            raise ValueError("No data loaded. Load data first.")
        
        # Get class distribution
        class_counts = {cls: len(frames) for cls, frames in self.frames.items()}
        print(f"Original class distribution: {class_counts}")
        
        if method == 'undersample':
            self._undersample_frames(random_state)
        elif method == 'oversample':
            self._oversample_frames(random_state)
        else:
            raise ValueError("method must be 'undersample' or 'oversample'")
        
        # Print new distribution
        new_counts = {cls: len(frames) for cls, frames in self.frames.items()}
        print(f"Balanced class distribution: {new_counts}")

    def _undersample_frames(self, random_state=None):
        """Undersample frames to match the minority class"""
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Find the minority class count
        class_counts = [len(frames) for frames in self.frames.values()]
        minority_count = min(class_counts)
        
        # Undersample each class
        balanced_frames = {}
        for class_name, frames in self.frames.items():
            if len(frames) > minority_count:
                # Undersample
                balanced_frames[class_name] = random.sample(frames, minority_count)
            else:
                # Keep all samples
                balanced_frames[class_name] = frames
        
        self.frames = balanced_frames

    def _oversample_frames(self, random_state=None):
        """Oversample frames to match the majority class"""
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Find the majority class count
        class_counts = [len(frames) for frames in self.frames.values()]
        majority_count = max(class_counts)
        
        # Oversample each class
        balanced_frames = {}
        for class_name, frames in self.frames.items():
            if len(frames) < majority_count:
                # Oversample with replacement
                balanced_frames[class_name] = resample(
                    frames, 
                    replace=True, 
                    n_samples=majority_count, 
                    random_state=random_state
                )
            else:
                # Keep all samples
                balanced_frames[class_name] = frames
        
        self.frames = balanced_frames

    # Also add these helper methods for data access:
    def get_data_dict(self):
        """Return the frames dictionary"""
        return self.frames.copy()

    def get_class_names(self):
        """Return the list of class names"""
        return self.class_names.copy()

    def get_total_samples(self):
        """Return total number of samples across all classes"""
        return sum(len(frames) for frames in self.frames.values())
