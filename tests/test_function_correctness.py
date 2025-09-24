# This script tests the correctness of the model-specific scripts' output.
# It verifies that the intermediate result files have the correct columns.
import os
import pandas as pd
import pytest

# Define the base directory relative to the test file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

@pytest.mark.parametrize("file_name", [
    'linear_model_results.csv', 
    'tree_model_results.csv', 
    'nn_model_results.csv', 
    'svm_model_results.csv'
])
def test_model_output_columns(file_name):
    """
    Tests if the model output CSVs have the correct required columns.
    """
    file_path = os.path.join(ARTIFACTS_DIR, file_name)
    
    # Skip test if the file doesn't exist (assuming previous scripts haven't been run)
    if not os.path.exists(file_path):
        pytest.skip(f"Artifact file '{file_name}' not found. Please run analysis scripts first.")
    
    df = pd.read_csv(file_path)
    required_columns = {'model_name', 'accuracy'}
    assert required_columns.issubset(set(df.columns)), \
        f"'{file_name}' is missing required columns. Expected: {required_columns}, Found: {set(df.columns)}."