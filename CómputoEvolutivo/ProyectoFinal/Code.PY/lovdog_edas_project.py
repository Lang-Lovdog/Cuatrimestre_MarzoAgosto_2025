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
    


if __name__ == "__main__":
#    df = pd.read_csv('./transformed_data.csv')
#    categoricalColumn = "target"
#    Values = {
#      "Graduate": 1,
#      "Dropout": 0
#    }
#    df[categoricalColumn] = df[categoricalColumn].map(Values)
#    model=ModelSelection("Seleccón", csv=df)
#    model.selectModels()
#    model.printSummary()
#    model.testModels()
#    model.printBestModel()
#    model.saveModels("models/")
#    model.saveConfusionMatrix()
#
#    # Initialize the EDA
#    eda = FeatureSelectionEDA(
#        models="./models",                 # Path to your models directory
#        dataset="./transformed_data.csv",  # Your dataset path or DataFrame
#        eID="exp_001",
#        nGenerations=200,
#        pSize=400,
#        ssSizeRange=(2, 12),               # Min 5, max 50, caractéristiques
#        eliteFraction=0.2,                 # Top 20% sont élite
#        mRate=0.1,                         # 10%, taux de mutation
#        classifierName="ensemble_eda",
#        targetName="target"                # Nom de la colonne de classification
#    )
#
#    # Check if models loaded properly
#    print(f"Number of models loaded: {len(eda.models)}")
#    print(f"Model paths: {eda.model_paths}")
#    print(f"Data shape: {eda.data.shape}")
#    print(f"Target shape: {eda.target.shape}")
#
#    # Run the algorithm
#    population = eda.run()
#
#    # Get results in different formats
#    # 1. As a pandas DataFrame (for analysis)
#    results_df = eda.get_results_dataframe()
#    print(results_df.head())
#
#    # 2. Export to CSV
#    eda.export_to_csv("my_eda_results.csv")
#
#    # 3. Export to Excel
#    eda.export_to_excel("my_eda_results.xlsx")
#
#    # 4. Get comprehensive final results
#    final_results = eda.get_final_results()
#    print(f"Best fitness: {final_results['best_fitness']}")
#    print(f"Selected features: {final_results['selected_features']}")
#
#    # 5. Export a detailed report
#    eda.export_final_report("experiment_final_report.txt")
# Step 1: Perform grid search and model selection
    model_selector = ModelSelection("SchoolDropout", csv="./transformed_data.csv")
    model_selector.selectModels()
    model_selector.testModels()
    
    # Get and display results
    summary_df = model_selector.getSummary()
    model_selector.saveSummary()
    print("Grid Search Results:")
    print(summary_df)

    
    # Save the best models
    model_selector.saveModels("./models")
    
    # Step 2: Use the best models for feature selection
    eda = FeatureSelectionEDA(
        model_selection_instance=model_selector,  # Pass the ModelSelection instance
        eID="feature_selection_exp",
        nGenerations=50,
        pSize=100,
        ssSizeRange=(5, 30),
        eliteFraction=0.2,
        mRate=0.1,
        classifierName="eda_feature_selection"
    )
    
    # Step 3: Run the feature selection
    population = eda.run()
    
    # Step 4: Get and analyze results
    results_df = eda.get_results_dataframe()
    final_results = eda.get_final_results()
    
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
