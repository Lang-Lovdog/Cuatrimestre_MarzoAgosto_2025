import numpy as np
from skimage.feature import graycomatrix, graycoprops
from LovdogFeatureBase import FeatureExtractor

class GLCMFeatures(FeatureExtractor):
    """
    GLCM (Gray-Level Co-occurrence Matrix) feature extractor.
    Extracts texture features using GLCM algorithm.
    """
    
    def __init__(self, distances=[1, 3, 7], angles=[0, 45, 90, 135], properties=None):
        super().__init__()
        self.distances = distances
        self.angles = angles
        self.properties = properties or ['contrast', 'correlation', 'energy', 'homogeneity']
        
        # Generate descriptive feature names
        self.feature_names_ = []
        for d in distances:
            for angle in angles:
                for prop in self.properties:
                    self.feature_names_.append(f"glcm_d{d}_a{angle}_{prop}")
    
    def fit_transform(self, data_dict):
        """Extract GLCM features from all images."""
        X_list, y_list = [], []
        
        for class_name, frames in data_dict.items():
            print(f"Extracting GLCM features for {class_name} ({len(frames)} images)...")
            for frame in frames:
                features = self._extract_single_image(frame)
                X_list.append(features)
                y_list.append(class_name)
        
        self.X_features_ = np.vstack(X_list)
        self.y_labels_ = np.array(y_list)
        return self.X_features_, self.y_labels_
    
    def _extract_single_image(self, image):
        """Extract GLCM features for a single image."""
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        features = []
        radians_angles = [np.radians(angle) for angle in self.angles]
        
        for d in self.distances:
            for angle_rad in radians_angles:
                glcm = graycomatrix(image, [d], [angle_rad], levels=256, 
                                  symmetric=True, normed=True)
                
                for prop in self.properties:
                    try:
                        feature_val = graycoprops(glcm, prop)[0, 0]
                        features.append(feature_val)
                    except Exception as e:
                        print(f"Warning: Could not compute {prop}: {e}")
                        features.append(0.0)
        
        return np.array(features)
