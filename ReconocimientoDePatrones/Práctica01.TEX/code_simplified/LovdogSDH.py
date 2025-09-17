import numpy                                      as np   # type: ignore
import pandas                                     as pd   # type: ignore
from   LovdogFeatureBase import FeatureExtractor          # type: ignore
from   typing            import Dict                      # type: ignore
from   typing            import List                      # type: ignore
from   typing            import Tuple                     # type: ignore

class SDHFeatures(FeatureExtractor):
    """
    Feature extractor based on Sum and Difference Histograms (SDH).
    Debugged and corrected based on C++ reference implementation.
    """

    def __init__(self, d: int = 1, angle: int = 0):
        super().__init__()
        self.d = d
        self.angle = angle
        self._validate_angle()

        # CORRECTED: Calculate dx, dy PROPERLY based on C++ implementation
        if self.angle == 0:
            self.dx = d
            self.dy = 0
        elif self.angle == 45:
            self.dx = d
            self.dy = d
        elif self.angle == 90:
            self.dx = 0
            self.dy = -d  # Note: C++ uses negative for 90/135
        elif self.angle == 135:
            self.dx = -d
            self.dy = -d

        # CORRECTED: Use the FULL set of features from C++ implementation
        self.feature_names_ = [
            "meanDiff", "meanSum", "varianceDiff", "varianceSum",
            "mean", "correlation", "contrast", "homogeneity",
            "energy", "entropy", "shadowness", "prominence"
        ]

    def _validate_angle(self):
        valid_angles = {0, 45, 90, 135}
        if self.angle not in valid_angles:
            raise ValueError(f"Angle must be one of {valid_angles}. Got {self.angle}.")

    def fit_transform(self, data_dict: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []

        for class_name, frame_list in data_dict.items():
            print(f"[SDH] Processing class: {class_name} ({len(frame_list)} frames)")

            for i, frame in enumerate(frame_list):
                # Prepare frame (convert to int16 like C++ CV_16SC1)
                frame_prepared = self._prepare_frame(frame).astype(np.int16)
                
                # Compute SDH features (following C++ logic exactly)
                features = self._compute_sdh_features(frame_prepared)
                
                X_list.append(features)
                y_list.append(class_name)

        self.X_features_ = np.vstack(X_list)
        self.y_labels_ = np.array(y_list)
        return self.X_features_, self.y_labels_

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize to 0-255 and ensure proper type"""
        frame_normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-6)
        return (frame_normalized * 255).astype(np.uint8)

    def _compute_sdh_features(self, img: np.ndarray) -> np.ndarray:
        """Main SDH computation following C++ logic exactly"""
        height, width = img.shape
        
        # CORRECTED: Proper region extraction based on C++ cv::Rect logic
        if self.angle == 0:
            # Right neighbor
            op1 = img[ 0        : height         , 0      : width-self.d ]
            op2 = img[ 0        : height         , self.d : width        ]
        elif self.angle == 45:
            # Diagonal down-right
            op1 = img[ self.d   : height         , 0      : width-self.d ]
            op2 = img[ 0        : height-self.d  , self.d : width        ]
        elif self.angle == 90:
            # Bottom neighbor (note: C++ uses negative dy, so we adjust indexing)
            op1 = img[self.d    : height         , 0      : width        ]
            op2 = img[ 0        : height-self.d  , 0      : width        ]
        elif self.angle == 135:
            # Diagonal down-left
            op1 = img[ self.d   : height         , self.d : width        ]
            op2 = img[ 0        : height-self.d  , 0      : width-self.d ]

        # Calculate sum and difference matrices
        sum_matrix  = op1 + op2
        diff_matrix = op1 - op2

        # Create histograms (EXACTLY like C++ code)
        sum_hist    = np.histogram( sum_matrix  , bins=511 , range=(   0, 510))[0]  #    0 to 510
        diff_hist   = np.histogram( diff_matrix , bins=511 , range=(-255, 255))[0]  # -255 to 255

        # Normalize histograms
        total_pixels_sum   = np.sum(sum_hist)
        total_pixels_diff  = np.sum(diff_hist)
        sum_hist_norm      = sum_hist  / total_pixels_sum
        diff_hist_norm     = diff_hist / total_pixels_diff

        # Calculate ALL features from C++ implementation
        return self._calculate_all_features(sum_hist_norm, diff_hist_norm)

    def _calculate_all_features(self, p_s: np.ndarray, p_d: np.ndarray) -> np.ndarray:
        """Calculate ALL 12 features from the C++ implementation"""
        bin_s        =  np.arange(511)      # 0 to 510
        bin_d        =  np.arange(511) - 255  # -255 to 255

        # 1. Mean of Difference
        mean_diff    =  np.sum(bin_d * p_d)
        
        # 2. Mean of Sum
        mean_sum     =  np.sum(bin_s * p_s)
        
        # 3. Variance of Difference
        var_diff     =  np.sum((bin_d - mean_diff)**2 * p_d)
        
        # 4. Variance of Sum
        var_sum      =  np.sum((bin_s - mean_sum)**2 * p_s)
        
        # 5. Overall Mean (average of sum mean)
        overall_mean = mean_sum / 2
        
        # 6. Correlation
        correlation  = (var_sum - var_diff) / (var_sum + var_diff + 1e-10)
        
        # 7. Contrast (variance of difference)
        contrast     =  var_diff
        
        # 8. Homogeneity
        homogeneity  =  np.sum(p_d / (1 + np.abs(bin_d)))
        
        # 9. Energy
        energy_sum   =  np.sum(p_s**2)
        energy_diff  =  np.sum(p_d**2)
        energy       =  energy_sum * energy_diff
        
        # 10. Entropy
        # Avoid log(0) by adding epsilon
        epsilon      =  1e-10
        entropy_sum  = -np.sum(p_s * np.log2(p_s + epsilon))
        entropy_diff = -np.sum(p_d * np.log2(p_d + epsilon))
        entropy      =  entropy_sum + entropy_diff
        
        # 11. Shadowness (Skewness of difference)
        shadowness   =  np.sum((bin_d - mean_diff)**3 * p_d)
        
        # 12. Prominence (Kurtosis of difference)
        prominence   =  np.sum((bin_d - mean_diff)**4 * p_d)

        # Return all 12 features as a vector
        return np.array([
            mean_diff, mean_sum, var_diff, var_sum,
            overall_mean, correlation, contrast, homogeneity,
            energy, entropy, shadowness, prominence
        ])

    def to_dataframe(self) -> pd.DataFrame:
        if self.X_features_ is None:
            raise ValueError("No features extracted. Call fit_transform first.")

        df = pd.DataFrame(self.X_features_, columns=self.feature_names_)
        df['class'] = self.y_labels_
        df['sdh_d'] = self.d
        df['sdh_angle'] = self.angle
        return df


class SDHFeaturesMultiAngle(FeatureExtractor):
    """Meta-extractor that computes SDH features for multiple angles."""
    
    def __init__(self, d: int = 1, angles: List[int] = [0, 45, 90, 135]):
        super().__init__()
        self.d = d
        self.angles = angles
        self.extractors = [SDHFeatures(d=d, angle=a) for a in angles]
        
        # Build combined feature names
        self.feature_names_ = []
        for angle in self.angles:
            angle_features = [f"{feat}_a{angle}" for feat in self.extractors[0].get_feature_names()]
            self.feature_names_.extend(angle_features)

    def fit_transform(self, data_dict):
        all_features = []
        
        for extractor in self.extractors:
            X_angle, y = extractor.fit_transform(data_dict)
            all_features.append(X_angle)
            if self.y_labels_ is None:
                self.y_labels_ = y

        self.X_features_ = np.hstack(all_features)
        return self.X_features_, self.y_labels_
