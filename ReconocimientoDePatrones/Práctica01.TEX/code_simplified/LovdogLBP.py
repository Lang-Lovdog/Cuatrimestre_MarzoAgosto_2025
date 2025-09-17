import numpy             as np                       #type: ignore
import pandas            as pd                       #type: ignore
from   skimage.feature   import graycoprops          #type: ignore
from   skimage.feature   import graycomatrix         #type: ignore
from   skimage.feature   import local_binary_pattern #type: ignore
from   LovdogFeatureBase import FeatureExtractor     #type: ignore
try:
    from typing import List
except ImportError:
    from types import List

class LBPFeatures(FeatureExtractor):

    def __init__(self, radius=1, n_points=8, method="uniform", window_sizes=[3]):
        super().__init__()
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.window_sizes = window_sizes if isinstance(window_sizes, list) else [window_sizes]
        
        # Define what our output columns will be. This defines feature_names_
        self.feature_names_ = [
            "mean",
            "variance",
            "correlation",
            "contrast",
            "homogeneity"
        ]
        # Note: "File" and "class" are not features, they are identifiers for the samples.

    def fit_transform(self, data_dict):
        """
        Computes LBP features from a LovdogDF data dictionary.
        """
        X_list = []  # Will hold the feature vectors for each sample
        y_list = []  # Will hold the label for each sample

        for class_name, frame_list in data_dict.items():
            print(f"[LBP] Processing class: {class_name}")
            
            for i, frame in enumerate(frame_list):
                # 1. Prepare the image (normalize to uint8)
                frame_normalized = self._normalize_frame(frame)
                
                # 2. Compute the LBP image
                lbp_image = local_binary_pattern(
                    frame_normalized, self.n_points, self.radius, self.method
                ).astype(np.uint8)

                # 3. Extract features from the LBP image using windowing
                sample_features = self._extract_features_from_lbp_image(lbp_image)
                
                # 4. Aggregate features for this sample (e.g., average across windows)
                #    This creates one feature vector per original input frame.
                if len(sample_features) > 0:
                    # Average the features from all windows for this frame
                    averaged_feature_vector = np.mean(sample_features, axis=0)
                    X_list.append(averaged_feature_vector)
                    y_list.append(class_name)

        # Convert lists to NumPy arrays
        self.X_features_ = np.vstack(X_list) # Stack list of vectors into a 2D matrix
        self.y_labels_ = np.array(y_list)    # Convert list of labels into a 1D array

        return self.X_features_, self.y_labels_

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize a frame to 0-255 uint8 range."""
        frame_normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-6)
        return (frame_normalized * 255).astype(np.uint8)

    def _extract_features_from_lbp_image(self, lbp_image: np.ndarray) -> List[np.ndarray]:
        """Extract features from an LBP image using sliding windows."""
        sample_features = []
        for ws in self.window_sizes:
            if ws < 0:  # Special case: use the whole image
                window = lbp_image
                features = self._compute_glcm_features(window)
                sample_features.append(features)
            else:
                # Sliding window logic
                for x in range(0, lbp_image.shape[0] - ws):
                    for y in range(0, lbp_image.shape[1] - ws):
                        window = lbp_image[x:x+ws, y:y+ws]
                        features = self._compute_glcm_features(window)
                        sample_features.append(features)
        return sample_features

    def _compute_glcm_features(self, window: np.ndarray) -> np.ndarray:
        """Compute GLCM features for a single window. Returns a 1D vector."""
        glcm = graycomatrix(window, [1], [0], symmetric=True, normed=True, levels=256)
        features = np.array([
            graycoprops(glcm, 'mean')[0, 0],
            graycoprops(glcm, 'variance')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0]
        ])
        return features

    # Optional: Method to get a DataFrame for visualization/export
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the extracted features to a Pandas DataFrame."""
        if self.X_features_ is None:
            raise ValueError("No features extracted. Call fit_transform first.")
        
        df = pd.DataFrame(self.X_features_, columns=self.get_feature_names())
        df['class'] = self.y_labels_
        return df
