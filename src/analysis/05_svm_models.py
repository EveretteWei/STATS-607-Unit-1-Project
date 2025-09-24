import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# --- Path Definitions ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# --- Load Processed Data ---
print("Loading processed data...")
try:
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data.csv'), header=None).values
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv'), header=None).values.ravel()
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'), header=None).values
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv'), header=None).values.ravel()
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Processed data not found. Please run the data processing script (01_...) first.")
    exit()

# This list will store the results from this script's models
model_results = []


# --- Define, Train, and Evaluate SVM Models ---
print("\n--- Training Support Vector Machine (SVM) Models ---")
# The parameters below are the best parameters found in the original code.
# We use them directly to avoid re-running a computationally expensive grid search.

# --- Model 11: Linear SVM ---
print("Training Linear SVM...")
linear_svm = SVC(kernel='linear', C=70, random_state=42)
linear_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)

model_results.append({'model_name': 'Linear SVM', 'accuracy': accuracy_linear})
print(f"  -> Test Accuracy: {accuracy_linear:.4f}")


# --- Model 12: Kernel SVM (RBF Kernel)---
print("Training RBF Kernel SVM...")
rbf_svm = SVC(kernel='rbf', C=70, gamma=0.01, random_state=42)
rbf_svm.fit(X_train, y_train)

y_pred_rbf = rbf_svm.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

model_results.append({'model_name': 'Kernel SVM (RBF)', 'accuracy': accuracy_rbf})
print(f"  -> Test Accuracy: {accuracy_rbf:.4f}")


# --- Model 13: Kernel SVM (Polynomial Kernel)---
print("Training Polynomial Kernel SVM...")
poly_svm = SVC(kernel='poly', C=50, degree=2, coef0=1.0, gamma='scale', random_state=42)
poly_svm.fit(X_train, y_train)

y_pred_poly = poly_svm.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)

model_results.append({'model_name': 'Kernel SVM (Poly)', 'accuracy': accuracy_poly})
print(f"  -> Test Accuracy: {accuracy_poly:.4f}")


# --- Save Results ---
print("\n--- Saving results for SVM models ---")
results_df = pd.DataFrame(model_results)
output_path = os.path.join(ARTIFACTS_DIR, 'svm_model_results.csv')
results_df.to_csv(output_path, index=False)
print(f"SVM model accuracies saved to: {output_path}")