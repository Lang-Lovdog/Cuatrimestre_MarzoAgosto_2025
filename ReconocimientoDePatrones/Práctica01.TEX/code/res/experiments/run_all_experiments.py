import sys
from pathlib import Path
import importlib.util
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_experiment(case_number):
    """Run a specific experiment case"""
    case_file = Path(__file__).parent / f"case{case_number}_glcm.py" if case_number == 1 else \
                Path(__file__).parent / f"case{case_number}_glr.py" if case_number == 2 else \
                Path(__file__).parent / f"case{case_number}_sdh.py" if case_number == 3 else \
                Path(__file__).parent / f"case{case_number}_combined.py" if case_number == 4 else \
                Path(__file__).parent / f"case{case_number}_best_lda.py"
    
    if not case_file.exists():
        print(f"❌ Case {case_number} file not found: {case_file}")
        return None
    
    try:
        # Dynamically import and run the case
        spec = importlib.util.spec_from_file_location(f"case{case_number}", case_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"\n{'='*80}")
        print(f"STARTING CASE {case_number}")
        print(f"{'='*80}")
        
        start_time = time.time()
        # Call the appropriate main function based on case number
        if case_number == 1:
            result = module.mainCase1()
        elif case_number == 2:
            result = module.mainCase2()
        elif case_number == 3:
            result = module.mainCase3()
        elif case_number == 4:
            result = module.mainCase4()
        elif case_number == 5:
            result = module.mainCase5()
            
        end_time = time.time()
        
        print(f"✅ Case {case_number} completed in {end_time - start_time:.2f} seconds")
        return result
        
    except Exception as e:
        print(f"❌ Error running Case {case_number}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all experiments in sequence"""
    cases_to_run = [1, 2, 3, 4, 5]
    results = {}
    
    for case_num in cases_to_run:
        result = run_experiment(case_num)
        if result:
            results[f"case{case_num}"] = result
            # Add a delay between cases to avoid file conflicts
            time.sleep(2)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    
    # Print summary
    for case, (model, metrics) in results.items():
        print(f"{case}: {metrics['best_classifier']} - {metrics['best_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    main()
