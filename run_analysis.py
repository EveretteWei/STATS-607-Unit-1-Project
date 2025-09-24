import subprocess
import sys
import os

def run_module(module_path):
    """
    Runs a Python module and checks for errors.
    """
    print(f"--- Running {os.path.basename(module_path)} ---")
    try:
        # Use runpy to execute the script as a module
        command = [sys.executable, '-m', module_path.replace(os.sep, '.')]
        # The script path should be relative to the project root
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"--- Successfully ran {os.path.basename(module_path)} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running {os.path.basename(module_path)}:")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        sys.exit(1) # Exit the entire process if a module fails

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    print("Starting the full project analysis pipeline...")

    # Define the order of modules to run
    pipeline_modules = [
        os.path.join('src', 'pipeline', '01_preprocess.py'),
        os.path.join('src', 'analysis', '02_linear_and_kernel_models.py'),
        os.path.join('src', 'analysis', '03_tree_based_models.py'),
        os.path.join('src', 'analysis', '04_neural_network.py'),
        os.path.join('src', 'analysis', '05_svm_models.py'),
        os.path.join('src', 'analysis', '06_summary_results.py')
    ]

    for module in pipeline_modules:
        if not os.path.exists(module):
            print(f"Error: Script not found at '{module}'. Please check your file paths.")
            sys.exit(1)
        # Convert path to module name (e.g., 'src/analysis/02_...' -> 'src.analysis.02_...')
        module_name = os.path.splitext(module)[0].replace(os.sep, '.')
        run_module(module_name)

    print("\nFull analysis pipeline completed successfully!")

if __name__ == "__main__":
    main()