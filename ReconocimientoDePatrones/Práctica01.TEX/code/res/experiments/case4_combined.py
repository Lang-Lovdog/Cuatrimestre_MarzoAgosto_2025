import numpy as np
from pathlib import Path

import sys
# Add parent directory to path to import Lovdog modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from res.experiments.base_experiment import BaseExperiment
from res.experiments.case1_glcm import Case1GLCMExperiment
from res.experiments.case2_glr import Case2GLRExperiment
from res.experiments.case3_sdh import Case3SDHExperiment

class Case4CombinedExperiment(BaseExperiment):
    """Case 4: Combined features with PCA/LDA."""
    
    def __init__(self):
        super().__init__("Case4_Combined")
    
    def extract_features(self, data_dict):
        """Combine features from all previous experiments."""
        # Run individual feature extractors
        glcm_exp = Case1GLCMExperiment()
        glr_exp = Case2GLRExperiment()
        sdh_exp = Case3SDHExperiment()
        
        X_glcm, y_glcm = glcm_exp.extract_features(data_dict)
        X_glr, y_glr = glr_exp.extract_features(data_dict)
        X_sdh, y_sdh = sdh_exp.extract_features(data_dict)
        
        # Verify all have same labels and order
        assert np.array_equal(y_glcm, y_glr) and np.array_equal(y_glcm, y_sdh)
        
        # Combine features horizontally
        X_combined = np.hstack([X_glcm, X_glr, X_sdh])
        
        return X_combined, y_glcm
    
    def run_complete(self):
        """Run with both PCA and LDA, return best results."""
        X, y = self.run()
        
        # With PCA
        ms_pca = self.run_classification(X, y, use_pca=True, n_components=0.95)
        acc_pca = ms_pca.results[ms_pca.best_model_tag]['test_accuracy']
        
        # With LDA
        ms_lda = self.run_classification(X, y, use_lda=True, n_components=None)
        acc_lda = ms_lda.results[ms_lda.best_model_tag]['test_accuracy']
        
        # Determine best case
        if acc_lda >= acc_pca:
            best_ms = ms_lda
            best_config = "with LDA"
            best_acc = acc_lda
            other_acc = acc_pca
        else:
            best_ms = ms_pca
            best_config = "with PCA"
            best_acc = acc_pca
            other_acc = acc_lda
        
        # Save best confusion matrix
        self.save_confusion_matrix(best_ms, "case4_combined_best_cm.png")
        
        results = {
            'experiment': 'Case4_Combined',
            'best_classifier': best_ms.best_model_tag,
            'best_configuration': best_config,
            'best_accuracy': best_acc,
            'other_accuracy': other_acc,
            'features_shape': X.shape,
            'pca_accuracy': acc_pca,
            'lda_accuracy': acc_lda
        }
        
        self.save_results(results, "case4_combined_results.csv")
        
        print(f"\nBest Combined result: {best_ms.best_model_tag} {best_config} - Accuracy: {best_acc:.4f}")
        
        return best_ms, results

def mainCase4():
    """Main function for Case 4."""
    experiment = Case4CombinedExperiment()
    return experiment.run_complete()

if __name__ == "__main__":
    mainCase4()
