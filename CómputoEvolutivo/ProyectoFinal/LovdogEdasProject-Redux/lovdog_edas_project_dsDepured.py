from lovdog_edas_project_h_dsDepured import ModelSelection
from lovdog_edas_project_h_dsDepured import FeatureSelectionEDA
import matplotlib.pyplot as plt #type: ignore
import numpy as np #type: ignore

# En supposant que la structure du projet est la suivante:
# ProyectoFinal/ ························· # Dossier du projet
# ├── models/ ···························· # Dossier où seront stockées des modèles
# ├── lovdog_edas_project.py ············· # Ce script (main script)
# ├── lovdog_edas_project_h.py ··········· # EDA header module
# ├── lovdog_edas_project_h_dsDepured.py · # Eda header module (version beta -depuré au DeepSeek-)
# └── transformed_data.csv ··············· # Fichier contenant les données
    


def main():
    # Step 1: Perform grid search and model selection
    model_selector = ModelSelection("SchoolDropout", csv="./transformed_data.csv")
    model_selector.selectModels()
    model_selector.testModels()
    
    # Get and display results
    summary_df = model_selector.getSummary()
    model_selector.saveSummary()
    model_selector.saveConfusionMatrix()
    print("Grid Search Results:")
    print(summary_df)

    
    # Save the best models
    model_selector.saveModels("./models")
    
    # Step 2: Use the best models for feature selection
    eda = FeatureSelectionEDA(
        model_selection_instance=model_selector,  # Pass the ModelSelection instance
        eID="feature_selection_exp",
        nGenerations=1250,
        pSize=500,
        ssSizeRange=(2, 15),
        eliteFraction=0.2,
        mRate=0.1,
        classifierName="eda_feature_selection"
    )
    
    # Step 3: Run the feature selection
    population = eda.run()
    
    # Step 4: Get and analyze results
    results_df = eda.get_results_dataframe()
    final_results = eda.get_final_results()
    results_df.to_csv("feature_selection_results.csv", index=False)
    
    print("\nFeature Selection Results:")
    print(f"Best fitness: {final_results['best_fitness']:.4f}")
    print(f"Selected {final_results['selected_feature_count']} features")
    print("Selected features:", final_results['selected_features'])
    
    # Export results
    eda.export_to_csv("feature_selection_results.csv")
    eda.export_final_report("feature_selection_report.txt")

    # After running both steps
    comparison_df = eda.compare_with_original_features(model_selector)
    print("\nPerformance Comparison:")
    print(comparison_df)

# Save comparison results
    comparison_df.to_csv("performance_comparison.csv", index=False)

# Plot the comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.35

    plt.bar(x - width/2, comparison_df['Original_Accuracy'], width, label='Original')
    plt.bar(x + width/2, comparison_df['EDA_Accuracy'], width, label='EDA Selected')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison: Original vs EDA-Selected Features')
    plt.xticks(x, comparison_df['Model'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
