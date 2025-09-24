# This script tests the integrity of the data processing pipeline.
# It verifies that the processed data files exist and are in the expected format.
import os
import pandas as pd
import sys

# Add the project root to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def test_processed_data_files():
    """
    Tests if the processed data files exist and have the correct format.
    """
    print("--- Running Data Validation Test ---")
    processed_data_dir = os.path.join(BASE_DIR, 'data', 'processed')
    
    # Check for file existence
    expected_files = ['train_data.csv', 'train_labels.csv', 'test_data.csv', 'test_labels.csv']
    for file_name in expected_files:
        file_path = os.path.join(processed_data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"FAILED: Expected processed file '{file_name}' not found.")
            return False
        
    print("PASSED: All processed data files exist.")

    # Check for correct dimensions (optional, but good practice)
    try:
        train_data = pd.read_csv(os.path.join(processed_data_dir, 'train_data.csv'), header=None)
        train_labels = pd.read_csv(os.path.join(processed_data_dir, 'train_labels.csv'), header=None)
        test_data = pd.read_csv(os.path.join(processed_data_dir, 'test_data.csv'), header=None)
        test_labels = pd.read_csv(os.path.join(processed_data_dir, 'test_labels.csv'), header=None)
        
        # Check if features and labels match
        if not (train_data.shape[0] == train_labels.shape[0] and test_data.shape[0] == test_labels.shape[0]):
            print("FAILED: Data and label files have mismatched row counts.")
            return False
            
    except Exception as e:
        print(f"FAILED: An error occurred while reading data files: {e}")
        return False
        
    print("PASSED: Processed data dimensions are consistent.")
    return True

if __name__ == "__main__":
    if test_processed_data_files():
        print("\nData validation test passed successfully!")
    else:
        sys.exit(1)