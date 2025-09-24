import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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


# --- Model 10: PyTorch Neural Network ---
# Train and Evaluate PyTorch MLP Model
print("\n--- Training PyTorch Neural Network ---")

# Prepare data for PyTorch
train_ds = TensorDataset(torch.from_numpy(X_train).float(), 
                         torch.from_numpy(y_train.astype(np.int64)))
test_ds = TensorDataset(torch.from_numpy(X_test).float(), 
                        torch.from_numpy(y_test.astype(np.int64)))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

pytorch_model = MLP(input_dim=X_train.shape[1], hidden_dim=100, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=1e-3)

n_epochs = 30
pytorch_model.train()
for epoch in range(n_epochs):
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = pytorch_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    # Print loss for each epoch
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_ds):.4f}")

# Evaluation loop
pytorch_model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = pytorch_model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)

y_pred_torch = np.concatenate(all_preds)
accuracy_torch = accuracy_score(y_test, y_pred_torch)
print(f"  -> Final Test Accuracy: {accuracy_torch:.4f}")


# --- Save Result ---
print("\n--- Saving neural network performance result ---")
result_df = pd.DataFrame([{'model_name': 'Neural Network', 'accuracy': accuracy_torch}])
output_path = os.path.join(ARTIFACTS_DIR, 'nn_model_results.csv')
result_df.to_csv(output_path, index=False)

print(f"Neural network training complete. Result saved to '{output_path}'.")