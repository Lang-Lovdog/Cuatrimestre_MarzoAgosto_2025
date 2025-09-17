import numpy  as     np              #type: ignore
from   abc    import abstractmethod  #type: ignore
from   abc    import ABC             #type: ignore
from   typing import Dict            #type: ignore
from   typing import List            #type: ignore
from   typing import Tuple           #type: ignore
from   typing import Any             #type: ignore
from   typing import Optional        #type: ignore

class FeatureExtractor(ABC):
    """
    Abstract Base Class for all feature extractors.
    Standardized interface for LovdogDF -> NumPy arrays.
    """

    def __init__(self):
        # We don't store the data_loader here anymore.
        # We just hold the extracted features and labels.
        self.X_features_ = None  # type: Optional[np.ndarray]
        self.y_labels_ = None    # type: Optional[np.ndarray]
        self.feature_names_ = [] # type: List[str] # Useful for interpretation

    @abstractmethod
    def fit_transform(self, data_dict: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to compute features from the data provided by LovdogDF.

        Args:
            data_dict: The dictionary from LovdogDF.get_data_dict(),
                    e.g., {'class_1': [np.array, np.array, ...], 'class_2': ...}

        Returns:
            Tuple (X, y):
                X: A 2D NumPy array of features (n_samples, n_features).
                y: A 1D NumPy array of labels (n_samples,).
        """
        pass

    def get_feature_names(self) -> List[str]:
        """
        Return a list of feature names. Crucial for understanding the model later.
        Must be implemented by the subclass.
        """
        return self.feature_names_

    def get_ml_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the extracted features and labels in the standard scikit-learn format.
        """
        if self.X_features_ is None or self.y_labels_ is None:
            raise ValueError("Features have not been extracted yet. Call fit_transform() first.")
        return self.X_features_, self.y_labels_
