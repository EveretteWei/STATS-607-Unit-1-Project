import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
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
    print("Error: Processed data not found. Please run '01_preprocess.py' first.")
    exit()

# This list will store the results from this script's models
model_results = []


# --- Define, Train, and Evaluate Tree-based Models ---
print("\n--- Training Tree-based and Ensemble Models ---")

models = {
# --- Model 5: CART ---
    'CART': DecisionTreeClassifier(random_state=42),
# --- Model 6: Bagging ---
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
# --- Model 7: Random Forest ---
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
# --- Model 8: AdaBoost ---
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
# --- Model 9: Gradient Boosting ---
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Loop through the dictionary to train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_results.append({'model_name': name, 'accuracy': accuracy})
    print(f"  -> Test Accuracy: {accuracy:.4f}")


# --- Save All Results ---
print("\n--- Saving results for tree-based models ---")
results_df = pd.DataFrame(model_results)
output_path = os.path.join(ARTIFACTS_DIR, 'tree_model_results.csv')
results_df.to_csv(output_path, index=False)

print(f"\nAll tree models trained and evaluated. Results saved to '{output_path}'.")