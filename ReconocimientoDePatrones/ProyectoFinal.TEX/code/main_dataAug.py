# main.py
from   sklearn.tree                  import DecisionTreeClassifier                  #type: ignore
from   sklearn.ensemble              import AdaBoostClassifier                      #type: ignore
from   sklearn.naive_bayes           import GaussianNB                              #type: ignore
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis              #type: ignore
#from sklearn.naive_bayes            import KernelNB                                 #type: ignore
# Note: KernelNB might not be available in all sklearn versions
# Alternative for Kernel Naive Bayes if not available:
#from   sklearn.naive_bayes           import GaussianNB                  as KernelNB #type: ignore # Fallback
from   sklearn.svm                   import SVC                                     #type: ignore
from   sklearn.neighbors             import KNeighborsClassifier                    #type: ignore
from   sklearn.linear_model          import LogisticRegression                      #type: ignore
from   sklearn.ensemble              import RandomForestClassifier                  #type: ignore
# Lovdog  Utils
from   LovdogData                    import load_file_with_extra                    #type: ignore
from   LovdogData                    import translate_px_aug                        #type: ignore
from   LovdogMS                      import ModelSelection                          #type: ignore
from   LovdogDF                      import LovdogDataFrames                        #type: ignore
from   LovdogLBP                     import LBPFeatures                             #type: ignore
import numpy                                                            as np       #type: ignore
import pandas                                                           as pd       #type: ignore
import os                                                                           #type: ignore
###############
###############
###############
###############
###############


def process_classifier_block(block_name, experiment_name, X, y, 
                           classifiers_dict, grid_params_dict, 
                           results_dir="results", models_dir="models", plots_dir="plots"):
    """
    Process a complete classifier block with all its classifiers.
    
    Parameters:
    -----------
    block_name : str
        Name of the classifier block (e.g., "Block1_Tree_Based")
    
    experiment_name : str
        Name of the experiment (e.g., "Experiment_I")
    
    X : array-like
        Feature matrix
    
    y : array-like
        Target labels
    
    classifiers_dict : dict
        Dictionary of classifiers for this block
    
    grid_params_dict : dict
        Dictionary of grid parameters for this block
    
    results_dir : str
        Directory to save results
    
    models_dir : str
        Directory to save models
    
    plots_dir : str
        Directory to save plots
    
    Returns:
    --------
    ModelSelection object and results metrics
    """
    
    # Create directories for this block
    block_dir = f"{block_name}_{experiment_name}"
    os.makedirs(f"{results_dir}/{block_dir}", exist_ok=True)
    os.makedirs(f"{models_dir}/{block_dir}", exist_ok=True)
    os.makedirs(f"{plots_dir}/{block_dir}", exist_ok=True)
    
    print(f"\nProcessing {block_name} - {experiment_name}")
    print("-" * 50)
    
    # Create ModelSelection instance for this block
    model_selector = ModelSelection(
        name=f"{block_name}_{experiment_name}",
        ml_data_tuple=(X, y),
        classifiers_dict=classifiers_dict,
        grid_parameters_dict=grid_params_dict,
        best_only_mode=True,
        plots_output_directory=f"{plots_dir}/{block_dir}",
        random_state=42
    )
    
    # Run complete analysis
    model_selector.selectModels()
    model_selector.generatePlots()
    
    # Get and save metrics
    metrics_df = model_selector.getMetricsDataFrame()
    metrics_df.to_csv(f"{results_dir}/{block_dir}/metrics.csv", index=False)
    
    # Save models
    model_selector.saveModels(f"{models_dir}/{block_dir}")
    
    # Get best classifier info
    best_classifier = model_selector.best_model_tag
    best_accuracy = metrics_df[metrics_df['Classifier'] == best_classifier]['Accuracy'].values[0]
    
    print(f"Best in {block_name}: {best_classifier} (Accuracy: {best_accuracy:.4f})")
    
    return model_selector, metrics_df


def get_classifier_blocks():
    """
    Return all four classifier blocks with their respective classifiers and parameters.
    
    Returns:
    --------
    dict: Dictionary containing all four blocks with their classifiers and grid parameters
    """
    
    # BLOCK 1: Tree-based classifiers
    block1 = {
        'classifiers': {
            'Decision_Tree': DecisionTreeClassifier(random_state=42),
            'Random_Forest': RandomForestClassifier(random_state=42),
            'KNN_Coarse': KNeighborsClassifier(n_neighbors=5),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        },
        'grid_params': {
            'Decision_Tree': {
                'clf__max_depth': [3, 5, 7, 10],
                'clf__min_samples_split': [2, 5, 10],
                'clf__criterion': ['gini', 'entropy']
            },
            'Random_Forest': {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [3, 5, 7, 10],
                'clf__min_samples_split': [2, 5, 10]
            },
            'KNN_Coarse': {
                'clf__n_neighbors': [10, 30, 50, 70, 90],
                'clf__weights': ['uniform', 'distance']
            },
            'AdaBoost': {
                'clf__n_estimators': [50, 100, 200],
                'clf__learning_rate': [0.01, 0.1, 1.0]
            }
        }
    }
    
    # BLOCK 2: KNN classifiers
    block2 = {
        'classifiers': {
            'KNN_Fine':      KNeighborsClassifier(n_neighbors=1),
            'KNN_Minkowski': KNeighborsClassifier(metric='minkowski'),
            'KNN_Weighted':  KNeighborsClassifier(weights='distance'),
            'KNN_Medium':    KNeighborsClassifier(n_neighbors=15),
        },
        'grid_params': {
            'KNN_Fine': {
                'clf__n_neighbors': [1, 2, 3],
                'clf__weights': ['uniform', 'distance']
            },
            'KNN_Minkowski': {
                'clf__n_neighbors': [5, 10, 15],
                'clf__p': [1, 2, 3],  # Minkowski distance parameter
                'clf__weights': ['uniform', 'distance']
            },
            'KNN_Weighted': {
                'clf__n_neighbors': [5, 10, 15],
                'clf__weights': ['distance']
            },
            'KNN_Medium': {
                'clf__n_neighbors': [10, 15, 20, 25],
                'clf__weights': ['uniform', 'distance']
            },
        }
    }
    
    # BLOCK 3: Probabilistic classifiers
    block3 = {
        'classifiers': {
            'Naive_Bayes_Gaussian': GaussianNB(),
#            'Naive_Bayes_Kernel': KernelNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'KNN_Cosine':    KNeighborsClassifier(metric='cosine')
        },
        'grid_params': {
            'Naive_Bayes_Gaussian': {
                'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
#            'Naive_Bayes_Kernel': {
#                # KernelNB might not have many parameters
#                'clf__kernel': ['gaussian', 'tophat', 'epanechnikov'] if hasattr(KernelNB, 'kernel') else []
#            },
            'LDA': {
                'clf__solver': ['svd', 'lsqr', 'eigen'],
                'clf__shrinkage': ['auto', 0.1, 0.5, 0.9]
            },
            'KNN_Cosine': {
                'clf__n_neighbors': [5, 10, 15],
                'clf__weights': ['uniform', 'distance']
            }
        }
    }
    
    # BLOCK 4: SVM classifiers
    block4 = {
        'classifiers': {
            'Linear_SVM':    SVC(kernel='linear', random_state=42),
            'Quadratic_SVM': SVC(kernel='poly', degree=2, random_state=42),
            'Cubic_SVM':     SVC(kernel='poly', degree=3, random_state=42),
            'Fifth_SVM':     SVC(kernel='poly', degree=5, random_state=42)
        },
        'grid_params': {
            'Linear_SVM': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__kernel': ['linear']
            },
            'Quadratic_SVM': {
                'clf__C': [0.1, 1, 10],
                'clf__degree': [2],
                'clf__gamma': ['scale', 'auto']
            },
            'Cubic_SVM': {
                'clf__C': [0.1, 1, 10],
                'clf__degree': [3],
                'clf__gamma': ['scale', 'auto']
            },
            'Fifth_SVM': {
                'clf__C': [0.1, 1, 10],
                'clf__degree': [5],
                'clf__gamma': ['scale', 'auto']
            }
        }
    }
    
    return {
        'Block1_Tree_Based': block1,
        'Block2_KNN_Variants': block2,
        'Block3_Probabilistic': block3,
        'Block4_SVM_Variants': block4
    }


def load_and_preprocess_data(config_path, experiment_name):
    """Load and preprocess data for an experiment"""
    print(f"\nLoading and balancing {experiment_name} data...")

    ## Create feature Matrix directory
    fm_dir = "feature_matrices"
    if not os.path.exists(fm_dir):
        os.makedirs(fm_dir)
    
    data_loader = LovdogDataFrames(config_path, load_file_with_extra)
    
    # Balance at the frame level (BEFORE feature extraction)
    print(f"   Total samples before balancing: {data_loader.get_total_samples()}")
    data_loader.balance_frames(method='undersample', random_state=42)
    print(f"   Total samples after balancing: {data_loader.get_total_samples()}")
    data_loader.augmentation_data(
        augmentation_function=translate_px_aug,
        extra_info={
            'offset_x': 6,
            'offset_y': 6,
            'tremble': True
        }
    )
    data_loader.augmentation_data(
        augmentation_function=translate_px_aug,
        extra_info={
            'offset_x': 2,
            'offset_y': 2,
            'tremble': True
        }
    )
    print(f"   Total samples after augmentation: {data_loader.get_total_samples()}")
    
    # Extract LBP features
    lbp_extractor = LBPFeatures(data_loader)
    lbp_extractor.set_parameters(window_size=-1)
    lbp_extractor.compute_lbp_from_data_loader()
    lbp_extractor.compute_features_from_lbp()
    lbp_extractor.save_to_csv(csv_path=f"{fm_dir}/{experiment_name}.csv")
    try:
        X, y = lbp_extractor.get_ml_data()
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    print(f"   {experiment_name} Features shape: {X.shape}")
    print(f"   Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return data_loader, X, y


def generate_final_summary(all_results):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    
    summary_data = []
    
    for result_key, result_data in all_results.items():
        metrics_df = result_data['metrics_df']
        best_row = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
        
        summary_data.append({
            'Block_Experiment': result_key,
            'Best_Classifier': best_row['Classifier'],
            'Accuracy': best_row['Accuracy'],
            'Precision': best_row['Precision'],
            'Recall': best_row['Recall'],
            'F1_Score': best_row['F1_Score'],
            'AUC_Score': best_row['AUC_Score'],
            'CV_Score': best_row['CV_Score']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("results/final_summary.csv", index=False)
    
    print(summary_df.to_string(index=False))
    
    # Find overall best performer
    best_overall = summary_df.loc[summary_df['Accuracy'].idxmax()]
    print(f"\n🏆 OVERALL BEST PERFORMER: {best_overall['Block_Experiment']}")
    print(f"   Best classifier: {best_overall['Best_Classifier']}")
    print(f"   Accuracy: {best_overall['Accuracy']:.4f}")
    print(f"   F1-Score: {best_overall['F1_Score']:.4f}")
    print(f"   AUC: {best_overall['AUC_Score']}")




def FinalStart():
    # 1. Load and preprocess data
    print("=" * 60)
    print("PATTERN RECOGNITION PROJECT - MAIN EXECUTION")
    print("=" * 60)
    
    # Load and balance data for both experiments
    data_loader_i,  X_i, y_i   = load_and_preprocess_data("data/config_experiment_i.json" , "Experiment_I")
    data_loader_ii, X_ii, y_ii = load_and_preprocess_data("data/config_experiment_ii.json", "Experiment_II")
    
    # 2. Get all classifier blocks
    classifier_blocks = get_classifier_blocks()
    
    # 3. Process all blocks for both experiments
    all_results = {}
    
    # Process Experiment I
    print("\n" + "=" * 60)
    print("PROCESSING EXPERIMENT I")
    print("=" * 60)
    
    for block_name, block_config in classifier_blocks.items():
        print(f"Processing {block_name} block...")
        model_selector, metrics_df = process_classifier_block(
            block_name=block_name,
            experiment_name="Experiment_I",
            X=X_i,
            y=y_i,
            classifiers_dict=block_config['classifiers'],
            grid_params_dict=block_config['grid_params']
        )
        print(f"Done with {block_name} block.")
        all_results[f"{block_name}_Experiment_I"] = {
            'model_selector': model_selector,
            'metrics_df': metrics_df
        }
    
    # Process Experiment II
    print("\n" + "=" * 60)
    print("PROCESSING EXPERIMENT II")
    print("=" * 60)
    
    for block_name, block_config in classifier_blocks.items():
        print(f"Processing {block_name} block...")
        model_selector, metrics_df = process_classifier_block(
            block_name=block_name,
            experiment_name="Experiment_II",
            X=X_ii,
            y=y_ii,
            classifiers_dict=block_config['classifiers'],
            grid_params_dict=block_config['grid_params']
        )
        print(f"Done with {block_name} block.")
        all_results[f"{block_name}_Experiment_II"] = {
            'model_selector': model_selector,
            'metrics_df': metrics_df
        }
    
    # 4. Generate final summary report
    generate_final_summary(all_results)
    
    print("\nAnalysis complete! All results saved.")


if __name__ == "__main__":
    FinalStart()
