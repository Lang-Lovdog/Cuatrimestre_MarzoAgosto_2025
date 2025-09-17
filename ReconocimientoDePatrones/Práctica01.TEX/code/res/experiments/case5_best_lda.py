import pandas as pd
from pathlib import Path

import sys
# Add parent directory to path to import Lovdog modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from res.experiments.base_experiment import BaseExperiment
from res.experiments.case1_glcm import Case1GLCMExperiment
from res.experiments.case2_glr import Case2GLRExperiment
from res.experiments.case3_sdh import Case3SDHExperiment

class Case5BestLDAExperiment(BaseExperiment):
    """Case 5: Apply LDA to best feature set from Cases 1-3."""
    
    def __init__(self):
        super().__init__("Case5_Best_LDA")
    
    def find_best_case(self):
        """Find the best performing case from 1-3."""
        results = []
        
        try:
            # Load results from previous cases
            case1_results = pd.read_csv("results/experiments/case1_glcm_results.csv")
            case2_results = pd.read_csv("results/experiments/case2_glr_results.csv")
            case3_results = pd.read_csv("results/experiments/case3_sdh_results.csv")
            
            results.extend([
                ('GLCM', case1_results.iloc[0]['best_accuracy'], 'case1'),
                ('GLR', case2_results.iloc[0]['best_accuracy'], 'case2'),
                ('SDH', case3_results.iloc[0]['best_accuracy'], 'case3')
            ])
            
        except FileNotFoundError:
            print("Previous case results not found. Running all cases...")
            # Run all cases if results not available
            _, _, case1_results = Case1GLCMExperiment().run_complete()
            _, case2_results = Case2GLRExperiment().run_complete()
            _, case3_results = Case3SDHExperiment().run_complete()
            
            results.extend([
                ('GLCM', case1_results['best_accuracy'], 'case1'),
                ('GLR', case2_results['best_accuracy'], 'case2'),
                ('SDH', case3_results['best_accuracy'], 'case3')
            ])
        
        # Find best case
        best_case = max(results, key=lambda x: x[1])
        print(f"Best case: {best_case[0]} with accuracy {best_case[1]:.4f}")
        
        return best_case
    
    def run_complete(self):
        """Run LDA on best feature set."""
        best_case_name, best_accuracy, case_id = self.find_best_case()
        
        # Get features from best case
        data_dict = self.load_data()
        
        if case_id == 'case1':
            extractor = Case1GLCMExperiment()
        elif case_id == 'case2':
            extractor = Case2GLRExperiment()
        else:  # case3
            extractor = Case3SDHExperiment()
        
        X, y = extractor.extract_features(data_dict)
        
        # Apply LDA - FIXED: use proper parameter passing
        ms_lda = self.run_classification(X, y, use_lda=True, n_components=None)
        acc_lda = ms_lda.results[ms_lda.best_model_tag]['test_accuracy']
        
        # Save confusion matrix
        self.save_confusion_matrix(ms_lda, f"case5_{case_id}_lda_cm.png")
        
        results = {
            'experiment': 'Case5_Best_LDA',
            'best_original_case': best_case_name,
            'best_original_accuracy': best_accuracy,
            'best_classifier': ms_lda.best_model_tag,
            'lda_accuracy': acc_lda,
            'improvement': acc_lda - best_accuracy,
            'features_shape': X.shape
        }
        
        self.save_results(results, "case5_best_lda_results.csv")
        
        print(f"\nLDA on {best_case_name}: {ms_lda.best_model_tag} - Accuracy: {acc_lda:.4f}")
        print(f"Improvement: {acc_lda - best_accuracy:+.4f}")
        
        return ms_lda, results

def mainCase5():
    """Main function for Case 5."""
    experiment = Case5BestLDAExperiment()
    return experiment.run_complete()

if __name__ == "__main__":
    mainCase5()
