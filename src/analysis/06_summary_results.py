import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns


# --- Path Definitions ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# Create results directories if they don't exist
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# --- Aggregate All Model Results ---
print("--- Aggregating model results from artifacts ---")
# Dynamically find all CSV files in the artifacts directory with a unified naming convention
csv_files = glob.glob(os.path.join(ARTIFACTS_DIR, '*_model_results.csv'))

if not csv_files:
    print("Error: No model accuracy CSV files found in the 'artifacts' directory.")
    print("Please ensure all model training scripts (02-05) have been run.")
    exit()

# Read and concatenate all found csv files
try:
    all_results_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
except Exception as e:
    print(f"Error reading and concatenating files: {e}")
    exit()

# Sort models by accuracy in descending order
all_results_df = all_results_df.sort_values(by='accuracy', ascending=False).reset_index(drop=True)


# --- Display and Save Final Table ---
print("\n--- Final Model Performance Summary ---")
print(all_results_df)

# Save the aggregated table to the results directory
final_table_path = os.path.join(TABLES_DIR, 'final_model_comparison.csv')
all_results_df.to_csv(final_table_path, index=False)
print(f"\nSummary table saved to: {final_table_path}")


# --- Generate and Save Visualization ---
print("\n--- Generating performance visualization ---")
plt.style.use('seaborn-v0_8-whitegrid') # Using a nice style for the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Create bar plot
sns.barplot(x='accuracy', y='model_name', data=all_results_df, ax=ax, palette='viridis')

# Add accuracy labels to the bars
for index, value in enumerate(all_results_df['accuracy']):
    ax.text(value + 0.005, index, f'{value:.4f}', va='center')

ax.set_title('Model Performance Comparison', fontsize=16)
ax.set_xlabel('Accuracy Score', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_xlim(0.9, 1.0) # Adjust x-axis to zoom in on the performance difference
plt.tight_layout()

# Save the figure
figure_path = os.path.join(FIGURES_DIR, 'model_performance_comparison.png')
plt.savefig(figure_path)
print(f"Performance chart saved to: {figure_path}")