import pandas as pd
import matplotlib.pyplot as plt

def generate_final_report():
    """Generate a comprehensive report of all experiments."""
    results_files = [
        "results/experiments/case1_glcm_results.csv",
        "results/experiments/case2_glr_results.csv",
        "results/experiments/case3_sdh_results.csv",
        "results/experiments/case4_combined_results.csv",
        "results/experiments/case5_best_lda_results.csv"
    ]
    
    all_results = []
    for file in results_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except FileNotFoundError:
            print(f"Warning: {file} not found")
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv("results/final_experiment_report.csv", index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        plt.bar(final_df['experiment'], final_df['best_accuracy'])
        plt.title('Comparison of All Experiments')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/experiment_comparison.png', dpi=300)
        
        print("Final report generated: results/final_experiment_report.csv")
    
    return final_df
