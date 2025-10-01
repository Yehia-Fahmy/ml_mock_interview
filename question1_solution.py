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
# SOLUTION:
pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / y_tr.sum()])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- TODO 2: Implement Training Loop ----
# SOLUTION:
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} | loss {loss.item():.4f}")

# ---- TODO 3: Implement Evaluation ----
# SOLUTION:
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_te))
    preds = (torch.sigmoid(logits) >= 0.5).int().numpy()
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary", zero_division=0)
    print(f"Precision {p:.3f} Recall {r:.3f} F1 {f1:.3f}")

# =============================================================================
# INSTRUCTOR HINTS FOR QUESTION 1: IMBALANCED CLASSIFICATION
# =============================================================================
# If the candidate gets stuck, here are progressive hints you can give:
#
# TODO 1 - Weighted Loss:
# Hint 1: "How do you tell PyTorch to weight certain examples more heavily?"
# Hint 2: "Look at the BCEWithLogitsLoss documentation - it has a pos_weight parameter"
# Hint 3: "Calculate pos_weight = (number of negative samples) / (number of positive samples)"
# Hint 4: "Convert to tensor: pos_weight = torch.tensor([ratio])"
#
# TODO 2 - Training Loop:
# Hint 1: "What's the typical PyTorch training loop structure?"
# Hint 2: "Don't forget to set model.train() and model.eval() modes"
# Hint 3: "Remember: zero_grad() → forward → loss → backward → step"
# Hint 4: "Use optimizer.zero_grad() before each backward pass"
#
# TODO 3 - Evaluation:
# Hint 1: "What happens if you don't use sigmoid on the raw logits?"
# Hint 2: "Set model.eval() and use torch.no_grad() for evaluation"
# Hint 3: "Convert logits to probabilities: torch.sigmoid(logits)"
# Hint 4: "Threshold at 0.5: (probabilities > 0.5).int()"
#
# General Hints:
# - "Why might accuracy be misleading for imbalanced data?"
# - "What's the difference between BCELoss and BCEWithLogitsLoss?"
# - "How does pos_weight affect the loss calculation?"
