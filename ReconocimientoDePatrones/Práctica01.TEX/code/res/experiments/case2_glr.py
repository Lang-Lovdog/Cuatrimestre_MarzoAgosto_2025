import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import Lovdog modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from LovdogGLRL import GLRFeatures
from res.experiments.base_experiment import BaseExperiment

class Case2GLRExperiment(BaseExperiment):
    """Case 2: GLR features with/without PCA."""
    
    def __init__(self):
        super().__init__("Case2_GLR")
    
    def extract_features(self, data_dict):
        """Extract GLR features for all images."""
        print("Extracting GLR features...")
        
        extractor = GLRFeatures(directions=[0, 45, 90])
        return extractor.fit_transform(data_dict)
    
    def run_complete(self):
        """Run both with and without PCA, return best results."""
        X, y = self.run()
        print(f"GLR features shape: {X.shape}")
        
        # Without PCA
        print("Training without PCA...")
        ms_no_pca = self.run_classification(X, y, use_pca=False)
        acc_no_pca = ms_no_pca.results[ms_no_pca.best_model_tag]['test_accuracy']
        print(f"Best without PCA: {ms_no_pca.best_model_tag} - Accuracy: {acc_no_pca:.4f}")
        
        # With PCA (keep 95% variance)
        print("Training with PCA...")
        ms_pca = self.run_classification(X, y, use_pca=True, n_components=0.95)
        acc_pca = ms_pca.results[ms_pca.best_model_tag]['test_accuracy']
        print(f"Best with PCA: {ms_pca.best_model_tag} - Accuracy: {acc_pca:.4f}")
        
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
        self.save_confusion_matrix(best_ms, "case2_glr_best_cm.png")
        
        results = {
            'experiment': 'Case2_GLR',
            'best_classifier': best_ms.best_model_tag,
            'best_configuration': best_config,
            'best_accuracy': best_acc,
            'features_shape': X.shape,
            'pca_accuracy': acc_pca,
            'no_pca_accuracy': acc_no_pca
        }
        
        self.save_results(results, "case2_glr_results.csv")
        
        print(f"\n✅ Best GLR result: {best_ms.best_model_tag} {best_config} - Accuracy: {best_acc:.4f}")
        
        return best_ms, results

def mainCase2():
    """Main function for Case 2."""
    experiment = Case2GLRExperiment()
    return experiment.run_complete()

if __name__ == "__main__":
    mainCase2()
