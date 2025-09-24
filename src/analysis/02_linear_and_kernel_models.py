import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Import the custom KernelLogisticRegression class from the new klr.py file
# This assumes klr.py is in a 'utils' folder one level above this script.
from klr import KernelLogisticRegression



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
    print("Error: Processed data not found. Please run '01_preprocess.py' first.")
    exit()

# This list will store the results
model_results = []


# --- Model 1: Penalized Logistic Regression ---
print("Training Penalized Logistic Regression...")
log_reg_model = LogisticRegressionCV(Cs=10**np.linspace(-4., 4., 20), 
                                     cv=5, 
                                     penalty='l2',
                                     solver='lbfgs', 
                                     max_iter=1000)
log_reg_model.fit(X_train, y_train)

y_pred_log_reg = log_reg_model.predict(X_test)
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)

model_results.append({'model_name': 'Penalized Logistic Regression', 'accuracy': acc_log_reg})
print(f"  -> Test Accuracy: {acc_log_reg:.4f}")


# --- Model 2: Linear Discriminant Analysis (LDA) ---
print("Training LDA...")
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

y_pred_lda = lda_model.predict(X_test)
acc_lda = accuracy_score(y_test, y_pred_lda)

model_results.append({'model_name': 'LDA', 'accuracy': acc_lda})
print(f"  -> Test Accuracy: {acc_lda:.4f}")


# --- Model 3: Kernel Logistic Regression (RBF Kernel)---
# We use the best parameters which are already found in the original code instead of cross-validation for simplicity.
print("Training Kernel Logistic Regression (RBF Kernel)...")
klr_rbf = KernelLogisticRegression(kernel='rbf', 
                                   gamma=0.2, 
                                   C=75.0)
# NOTE: The custom KernelLogisticRegression may raise a ConvergenceWarning here.
# This indicates the optimizer did not fully converge within the specified number of iterations.
# The warning is intentionally left visible to transparently document the model's 
# optimization behavior with these specific parameters. The resulting accuracy is still captured for comparison.
klr_rbf.fit(X_train, y_train)

y_pred_klr_rbf = klr_rbf.predict(X_test)
acc_klr_rbf = accuracy_score(y_test, y_pred_klr_rbf)

model_results.append({'model_name': 'Kernel Logistic Regression (RBF)', 'accuracy': acc_klr_rbf})
print(f"  -> Test Accuracy: {acc_klr_rbf:.4f}")


# --- Model 4: Kernel Logistic Regression (Polynomial Kernel) ---
# We use the best parameters which are already found in the original code instead of cross-validation for simplicity.
print("Training Kernel Logistic Regression (Polynomial Kernel)...")
klr_poly = KernelLogisticRegression(kernel='poly', 
                                    degree=2, 
                                    C=0.01, 
                                    coef0=100)
# NOTE: The custom KernelLogisticRegression may raise a ConvergenceWarning here.
# This indicates the optimizer did not fully converge within the specified number of iterations.
# The warning is intentionally left visible to transparently document the model's 
# optimization behavior with these specific parameters. The resulting accuracy is still captured for comparison.
klr_poly.fit(X_train, y_train)

y_pred_klr_poly = klr_poly.predict(X_test)
acc_klr_poly = accuracy_score(y_test, y_pred_klr_poly)

model_results.append({'model_name': 'Kernel Logistic Regression (Poly)', 'accuracy': acc_klr_poly})
print(f"  -> Test Accuracy: {acc_klr_poly:.4f}")


# --- Save Results ---
print("\n--- Saving results for linear models ---")
results_df = pd.DataFrame(model_results)
output_path = os.path.join(ARTIFACTS_DIR, 'linear_model_results.csv')
results_df.to_csv(output_path, index=False)
print(f"Results saved to '{output_path}'.")