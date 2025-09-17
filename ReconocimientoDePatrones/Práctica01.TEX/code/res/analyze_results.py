"""
Analyze and compare results from different feature extractors
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob

def analyze_results():
    """Analyze and compare results from all experiments."""
    
    # Load all result files
    result_files = glob.glob("results/*_metrics.csv")
    
    if not result_files:
        print("No result files found. Run main.py first.")
        return
    
    # Combine results
    all_results = []
    for file in result_files:
        df = pd.read_csv(file)
        df['extractor'] = file.split('/')[-1].replace('_metrics.csv', '')
        all_results.append(df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        for extractor in combined_df['extractor'].unique():
            extractor_data = combined_df[combined_df['extractor'] == extractor]
            plt.bar(extractor, extractor_data[metric].values[0], label=extractor)
        
        plt.title(f'{metric} Comparison')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary = combined_df.groupby('extractor')[metrics].mean()
    summary.to_csv('results/summary_comparison.csv')
    
    print("Results analysis completed!")
    print("\nBest performing extractors:")
    for metric in metrics:
        best_extractor = summary[metric].idxmax()
        best_value = summary[metric].max()
        print(f"  {metric}: {best_extractor} ({best_value:.3f})")

if __name__ == "__main__":
    analyze_results()
