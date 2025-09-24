# This script tests the correctness of the model-specific scripts' output.
# It verifies that the intermediate result files have the correct format (e.g., expected columns).
import os
import pandas as pd
import sys

# Add the project root to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def test_model_output_format():
    """
    Tests if the model output CSVs have the correct columns.
    """
    print("--- Running Function Correctness Test (Model Output) ---")
    artifacts_dir = os.path.join(BASE_DIR, 'artifacts')
    
    # Assumes all your model scripts save to a consistent *_results.csv format
    # except for the SVM one, which we'll handle below for robustness.
    model_files = [
        'linear_model_results.csv', 
        'tree_model_results.csv', 
        'nn_model_results.csv', 
        'svm_model_accuracies.csv'
    ]
    
    required_columns = {'model_name', 'accuracy'}
    
    for file_name in model_files:
        file_path = os.path.join(artifacts_dir, file_name)
        if not os.path.exists(file_path):
            print(f"SKIPPED: Artifact file '{file_name}' not found. Please run analysis scripts first.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if not required_columns.issubset(df.columns):
                print(f"FAILED: '{file_name}' is missing required columns. Expected: {required_columns}, Found: {set(df.columns)}")
                return False
            else:
                print(f"PASSED: '{file_name}' has the correct output format.")
        except Exception as e:
            print(f"FAILED: An error occurred while reading '{file_name}': {e}")
            return False
            
    return True

if __name__ == "__main__":
    if test_model_output_format():
        print("\nFunction correctness test passed successfully!")
    else:
        sys.exit(1)