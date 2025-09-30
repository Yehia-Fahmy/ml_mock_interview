import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# ---- Synthetic Data ----
def make_synthetic(n=5000, pos_ratio=0.2, d=10):
    """Create imbalanced synthetic dataset (20% positive class)"""
    n_pos = int(n * pos_ratio)
    n_neg = n - n_pos
    X_neg = np.random.normal(0, 1.0, size=(n_neg, d))
    X_pos = np.random.normal(1.0, 1.0, size=(n_pos, d))
    X = np.vstack([X_neg, X_pos]).astype(np.float32)
    y = np.hstack([np.zeros(n_neg), np.ones(n_pos)]).astype(np.float32)
    perm = np.random.permutation(n)
    return X[perm], y[perm]

X, y = make_synthetic()
split = int(0.8 * len(X))
X_tr, y_tr = X[:split], y[:split]
X_te, y_te = X[split:], y[split:]

train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
test_ds  = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

# ---- Model ----
class MLP(nn.Module):
    def __init__(self, d=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.fc(x).squeeze(1)  # logits

model = MLP().to("cpu")

# ---- TODO 1: Implement Weighted BCE Loss ----
# The dataset is imbalanced (20% positive, 80% negative).
# Task: Create a loss function that handles this class imbalance.
# Hint: Use BCEWithLogitsLoss with pos_weight parameter.
# The weight should be inversely proportional to class frequency.
# Formula: pos_weight = (number of negative samples) / (number of positive samples)

# YOUR CODE HERE:
criterion = None  # Replace with weighted BCE loss

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- TODO 2: Implement Training Loop ----
# Task: Train the model for 5 epochs
# For each epoch:
#   1. Set model to training mode
#   2. Loop through batches in train_loader
#   3. Zero gradients, compute loss, backpropagate, update weights
#   4. Print the loss at the end of each epoch

# YOUR CODE HERE:
for epoch in range(5):
    pass  # Implement training loop


# ---- TODO 3: Implement Evaluation ----
# Task: Evaluate the model on test data
# Steps:
#   1. Set model to evaluation mode
#   2. Disable gradient computation
#   3. Get predictions on X_te (convert logits to probabilities with sigmoid, threshold at 0.5)
#   4. Compute and print Precision, Recall, and F1 score
# Hint: Use precision_recall_fscore_support from sklearn.metrics

# YOUR CODE HERE:
# (Implement evaluation)


# ---- Expected Output ----
# After successful implementation, you should see:
# - Training loss decreasing over 5 epochs
# - Final F1 score around 0.75-0.85
# - High recall (0.9+) indicating the model catches most positive cases
