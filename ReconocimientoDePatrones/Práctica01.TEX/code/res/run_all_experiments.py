"""
Run all experiments sequentially.
"""

import time
from .experiments.case1_glcm import mainCase1
from .experiments.case2_glr import mainCase2
from .experiments.case3_sdh import mainCase3
from .experiments.case4_combined import mainCase4
from .experiments.case5_best_lda import mainCase5

import sys
import os

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_experiments():
    """Run all experiments in sequence."""
    start_time = time.time()
    
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    
    # Run each case
    experiments = [
        ("Case 1: GLCM", mainCase1),
        ("Case 2: GLR", mainCase2),
        ("Case 3: SDH", mainCase3),
        ("Case 4: Combined", mainCase4),
        ("Case 5: Best + LDA", mainCase5)
    ]
    
    results = {}
    
    for name, experiment_func in experiments:
        print(f"\n{name}")
        print("-" * 50)
        try:
            result = experiment_func()
            results[name] = result
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    print("All experiments completed! 🎉")
    
    return results

if __name__ == "__main__":
    run_all_experiments()
