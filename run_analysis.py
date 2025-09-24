import subprocess
import sys
import os

def run_script(script_path):
    """
    Runs a Python script and checks for errors.
    """
    print(f"--- Running {os.path.basename(script_path)} ---")
    try:
        # Use subprocess to run the script directly
        result = subprocess.run([sys.executable, script_path], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"--- Successfully ran {os.path.basename(script_path)} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running {os.path.basename(script_path)}:")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        sys.exit(1) # Exit the entire process if a module fails

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    print("Starting the full project analysis pipeline...")

    # Define the order of scripts to run
    # Note: 01_preprocess.py should be in src/pipeline/
    # All other scripts are assumed to be in src/analysis/
    pipeline_scripts = [
        os.path.join('src', 'pipeline', '01_preprocess.py'),
        os.path.join('src', 'analysis', '02_linear_and_kernel_models.py'),
        os.path.join('src', 'analysis', '03_tree_based_models.py'),
        os.path.join('src', 'analysis', '04_neural_network.py'),
        os.path.join('src', 'analysis', '05_svm_models.py'),
        os.path.join('src', 'analysis', '06_summary_results.py')
    ]

    for script in pipeline_scripts:
        if not os.path.exists(script):
            print(f"Error: Script not found at '{script}'. Please check your file paths.")
            sys.exit(1)
        run_script(script)

    print("\nFull analysis pipeline completed successfully!")

if __name__ == "__main__":
    main()