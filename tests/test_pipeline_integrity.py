# This script tests the end-to-end integrity of the project pipeline.
# It verifies that the final figures and tables are successfully generated.
import os
import sys

# Add the project root to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def test_final_outputs_exist():
    """
    Tests if the final figures and tables were generated.
    """
    print("--- Running Pipeline Integrity Test ---")
    
    all_passed = True
    
    # Check for final table
    final_table_path = os.path.join(BASE_DIR, 'results', 'tables', 'final_model_comparison.csv')
    if not os.path.exists(final_table_path):
        print("FAILED: Final results table not found at 'results/tables/final_model_comparison.csv'.")
        all_passed = False
    else:
        print("PASSED: Final results table found.")

    # Check for final figure
    final_figure_path = os.path.join(BASE_DIR, 'results', 'figures', 'model_performance_comparison.png')
    if not os.path.exists(final_figure_path):
        print("FAILED: Final performance plot not found at 'results/figures/model_performance_comparison.png'.")
        all_passed = False
    else:
        print("PASSED: Final performance plot found.")
        
    return all_passed

if __name__ == "__main__":
    if test_final_outputs_exist():
        print("\nPipeline integrity test passed successfully!")
    else:
        sys.exit(1)