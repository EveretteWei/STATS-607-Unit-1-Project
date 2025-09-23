import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


# --- 1. Define File Paths ---
# Use relative paths to ensure the script runs on any machine.
# __file__ is the path to the current script. We go up two directories to get to the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Input file paths
TRAIN_INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'spam-train.txt')
TEST_INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'spam-test.txt')

# Output directory path
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


# --- 2. Create Output Directory ---
# Create the 'data/processed' directory if it doesn't exist.
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
print(f"Output directory '{PROCESSED_DATA_DIR}' is ready.")


# --- 3. Load Raw Data ---
print("Loading raw data...")
try:
    train_datatot = pd.read_csv(TRAIN_INPUT_PATH, header=None, sep=',')
    test_datatot = pd.read_csv(TEST_INPUT_PATH, header=None, sep=',')
    print("Raw data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Data files not found at '{TRAIN_INPUT_PATH}' or '{TEST_INPUT_PATH}'.")
    print("Please ensure spam-train.txt and spam-test.txt are in the 'data/raw/spam-data/' folder.")
    exit()


# --- 4. Separate Features and Labels ---
train_data = train_datatot.iloc[:, :-1]
train_labels = train_datatot.iloc[:, -1]
test_data = test_datatot.iloc[:, :-1]
test_labels = test_datatot.iloc[:, -1]
print("Features and labels separated.")


# --- 5. Standardize the Data ---
print("Standardizing the data...")
scaler = StandardScaler()

# Fit the scaler on the training data and transform it.
train_data_standardized = scaler.fit_transform(train_data)
# Use the same scaler to transform the test data for consistency.
test_data_standardized = scaler.transform(test_data)
print("Data standardization complete.")


# --- 6. Save Processed Data ---
print("Saving processed files to data/processed/ ...")
# We save features and labels separately for easier use later.
np.savetxt(os.path.join(PROCESSED_DATA_DIR, 'train_data.csv'), train_data_standardized, delimiter=',')
np.savetxt(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'), test_data_standardized, delimiter=',')
train_labels.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv'), header=False, index=False)
test_labels.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv'), header=False, index=False)

print("\nData preprocessing complete!")
print(f"Four output files have been saved to the '{PROCESSED_DATA_DIR}' folder.")