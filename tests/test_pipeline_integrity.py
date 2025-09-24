# This script tests the end-to-end integrity of the project pipeline.
# It verifies that the final figures and tables are successfully generated.
import os
import pytest

# Define the base directory relative to the test file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FIGURES_DIR = os.path.join(BASE_DIR, 'results', 'figures')
RESULTS_TABLES_DIR = os.path.join(BASE_DIR, 'results', 'tables')

@pytest.mark.parametrize("file_path", [
    os.path.join(RESULTS_TABLES_DIR, 'final_model_comparison.csv'),
    os.path.join(RESULTS_FIGURES_DIR, 'model_performance_comparison.png')
])
def test_final_outputs_exist(file_path):
    """
    Tests if the final figures and tables were generated.
    """
    assert os.path.exists(file_path), f"Final output file not found at '{file_path}'."