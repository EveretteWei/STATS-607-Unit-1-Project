# This script tests the integrity of the data processing pipeline using pytest.
# It verifies that the processed data files exist and are in the expected format.
import os
import pandas as pd
import pytest

# Define the base directory relative to the test file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

@pytest.mark.parametrize("file_name", [
    'train_data.csv', 'train_labels.csv', 'test_data.csv', 'test_labels.csv'
])
def test_processed_data_files_exist(file_name):
    """
    Tests if processed data files exist in the expected directory.
    """
    file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    assert os.path.exists(file_path), f"Processed file '{file_name}' not found at '{file_path}'."

def test_processed_data_dimensions():
    """
    Tests if the processed data and label files have matching dimensions.
    """
    train_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data.csv'), header=None)
    train_labels = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv'), header=None)
    test_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'), header=None)
    test_labels = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv'), header=None)
    
    assert train_data.shape[0] == train_labels.shape[0], "Train data and labels have mismatched row counts."
    assert test_data.shape[0] == test_labels.shape[0], "Test data and labels have mismatched row counts."