import numpy as np
from .base_experiment import BaseExperiment

import sys
from pathlib import Path

# Add parent directory to path to import Lovdog modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from LovdogSDH import SDHFeatures, SDHFeaturesMultiAngle

class Case3SDHExperiment(BaseExperiment):
    """Case 3: SDH features with/without PCA."""
    
    def __init__(self):
        super().__init__("Case3_SDH")
    
    def extract_features(self, data_dict):
        """Extract SDH features for all images."""
        # Use multi-angle SDH for comprehensive features
        sdh_extractor = SDHFeaturesMultiAngle(d=1, angles=[0, 45, 90, 135])
        return sdh_extractor.fit_transform(data_dict)
    
    def run_complete(self):
        """Run both with and without PCA, return best results."""
        X, y = self.run()
        
        # Without PCA
        ms_no_pca = self.run_classification(X, y, use_pca=False)
        acc_no_pca = ms_no_pca.results[ms_no_pca.best_model_tag]['test_accuracy']
        
        # With PCA
        ms_pca = self.run_classification(X, y, use_pca=True, n_components=0.95)
        acc_pca = ms_pca.results[ms_pca.best_model_tag]['test_accuracy']
        
        # Determine best case
        if acc_pca >= acc_no_pca:
            best_ms = ms_pca
            best_config = "with PCA"
            best_acc = acc_pca
        else:
            best_ms = ms_no_pca
            best_config = "without PCA"
            best_acc = acc_no_pca
        
        # Save best confusion matrix
        self.save_confusion_matrix(best_ms, "case3_sdh_best_cm.png")
        
        results = {
            'experiment': 'Case3_SDH',
            'best_classifier': best_ms.best_model_tag,
            'best_configuration': best_config,
            'best_accuracy': best_acc,
            'features_shape': X.shape,
            'pca_accuracy': acc_pca,
            'no_pca_accuracy': acc_no_pca
        }
        
        self.save_results(results, "case3_sdh_results.csv")
        
        print(f"\nBest SDH result: {best_ms.best_model_tag} {best_config} - Accuracy: {best_acc:.4f}")
        
        return best_ms, results

def mainCase3():
    """Main function for Case 3."""
    experiment = Case3SDHExperiment()
    return experiment.run_complete()

if __name__ == "__main__":
    mainCase3()
