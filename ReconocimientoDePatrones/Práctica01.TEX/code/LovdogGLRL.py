import numpy as np
from LovdogFeatureBase import FeatureExtractor

class GLRFeatures(FeatureExtractor):
    """
    GLR (Gray-Level Run-Length) feature extractor.
    Extracts run-length based texture features.
    """
    
    def __init__(self, directions=[0, 45, 90]):
        super().__init__()
        self.directions = directions
        
        # Feature names for each direction
        self.feature_names_ = []
        feature_types = ['SRE', 'LRE', 'GLNU', 'RLN', 'RP']
        for d in directions:
            for ft in feature_types:
                self.feature_names_.append(f"glrl_d{d}_{ft}")
    
    def fit_transform(self, data_dict):
        """Extract GLR features from all images."""
        X_list, y_list = [], []
        
        for class_name, frames in data_dict.items():
            print(f"Extracting GLR features for {class_name} ({len(frames)} images)...")
            for frame in frames:
                features = self._extract_single_image(frame)
                X_list.append(features)
                y_list.append(class_name)
        
        self.X_features_ = np.vstack(X_list)
        self.y_labels_ = np.array(y_list)
        return self.X_features_, self.y_labels_
    
    def _extract_single_image(self, image):
        """Extract GLR features for a single image."""
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        features = []
        for direction in self.directions:
            glrlm = self._compute_glrlm(image, direction)
            dir_features = self._compute_glrlm_features(glrlm)
            features.extend(dir_features)
        
        return np.array(features)
    
    def _compute_glrlm(self, image, direction, levels=256):
        """Compute Gray-Level Run-Length Matrix for given direction."""
        if direction == 0:    # Horizontal
            working_image = image
        elif direction == 45: # Diagonal
            working_image = np.fliplr(image)
        elif direction == 90: # Vertical
            working_image = image.T
        else:
            raise ValueError("Direction must be 0, 45, or 90 degrees")
        
        rows, cols = working_image.shape
        max_run_length = max(rows, cols)
        glrlm = np.zeros((levels, max_run_length), dtype=np.int32)
        
        # Process each row
        for row in working_image:
            run_length = 1
            for i in range(1, len(row)):
                if row[i] == row[i - 1]:
                    run_length += 1
                else:
                    gray_level = row[i - 1]
                    if gray_level < levels and run_length <= max_run_length:
                        glrlm[gray_level, run_length - 1] += 1
                    run_length = 1
            
            # Handle last run in the row
            gray_level = row[-1]
            if gray_level < levels and run_length <= max_run_length:
                glrlm[gray_level, run_length - 1] += 1
        
        return glrlm
    
    def _compute_glrlm_features(self, glrlm):
        """Compute features from GLRLM."""
        eps = 1e-12
        total_runs = np.sum(glrlm)
        
        if total_runs == 0:
            return [0.0] * 5  # Return zeros for empty matrix
        
        j_values = np.arange(1, glrlm.shape[1] + 1)
        
        # Short Run Emphasis (SRE)
        SRE = np.sum(glrlm / (j_values ** 2 + eps)) / (total_runs + eps)
        
        # Long Run Emphasis (LRE)
        LRE = np.sum(glrlm * (j_values ** 2)) / (total_runs + eps)
        
        # Gray-Level Non-Uniformity (GLNU)
        GLNU = np.sum(np.sum(glrlm, axis=1) ** 2) / (total_runs + eps)
        
        # Run Length Non-Uniformity (RLN)
        RLN = np.sum(np.sum(glrlm, axis=0) ** 2) / (total_runs + eps)
        
        # Run Percentage (RP)
        RP = total_runs / (glrlm.shape[0] * glrlm.shape[1] + eps)
        
        return [SRE, LRE, GLNU, RLN, RP]
