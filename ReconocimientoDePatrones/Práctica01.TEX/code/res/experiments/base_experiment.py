import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os
from datetime import datetime
import joblib  # Add this import

# Add the project root to the system path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from LovdogMS import ModelSelection

class BaseExperiment:
    """Base class for all experiments with common functionality."""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        
        # Set up paths based on your actual structure
        self.project_root = Path(__file__).parent.parent.parent
        self.output_dir = self.project_root / "results" / "experiments" / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        self.results = {}
        
    def load_data(self):
        """Load data using LovdogDF."""
        from LovdogDF import LovdogDataFrames
        # Use the correct path to images.json
        json_path = self.project_root / "res" / "imagenes" / "images.json"
        print(f"Loading data from: {json_path}")
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found at: {json_path}")
            
        data_porter = LovdogDataFrames(str(json_path))
        return data_porter.get_feature_extractor_input()
    
    def extract_features(self, data_dict):
        """Extract features - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement extract_features")
    
    def run_classification(self, X, y, use_pca=False, use_lda=False, n_components=None):
        """Run classification with optional dimensionality reduction."""
        # Create unique name with timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"{'PCA' if use_pca else 'LDA' if use_lda else 'Base'}"
        name = f"{self.experiment_name}_{config_name}_{timestamp}"
        
        ms = ModelSelection(
            name=name,
            ml_data_tuple=(X, y),
            best_only_mode=True,
            plots_output_directory=str(self.output_dir / "plots")
        )
        
        # Run selection with the specified dimensionality reduction
        ms.selectModels(use_pca=use_pca, use_lda=use_lda, n_components=n_components)
        return ms
    
    def save_confusion_matrix(self, ms, filename):
        """Save confusion matrix for the best model."""
        # Ensure detailed results are generated
        if not hasattr(ms, 'detailed_results') or not ms.detailed_results:
            ms._generateDetailedResultsForBest()
        
        cm = ms.detailed_results["confusion_matrix"]
        
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {ms.best_model_tag}\n{self.experiment_name}')
        plt.savefig(self.output_dir / "plots" / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def save_results(self, results_dict, filename):
        """Save results to CSV with unique naming."""
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        unique_filename = f"{base_name}_{timestamp}{extension}"
        
        df = pd.DataFrame([results_dict])
        df.to_csv(self.output_dir / "metrics" / unique_filename, index=False)
        # Also save a latest version without timestamp
        df.to_csv(self.output_dir / "metrics" / filename, index=False)
    
    def save_model(self, ms, filename):
        """Save the best model to disk."""
        joblib.dump(ms.best_model, self.output_dir / "models" / filename)
    
    def run(self):
        """Main experiment workflow."""
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {self.experiment_name}")
        print(f"{'='*60}")
        
        # Load data
        data_dict = self.load_data()
        print("✓ Data loaded successfully")
        
        # Extract features
        X, y = self.extract_features(data_dict)
        print(f"✓ Features extracted: {X.shape}")
        
        return X, y
