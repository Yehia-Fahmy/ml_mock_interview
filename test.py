import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# ---- Synthetic Data ----
def make_synthetic(n=5000, pos_ratio=0.2, d=10):
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

# ---- TODO: Weighted BCE Loss ----
# How do you tell PyTorch to weight certain examples more heavily?
pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / y_tr.sum()])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- Training Loop ----
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} | loss {loss.item():.4f}")

# ---- Evaluation ----
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_te))
    preds = (torch.sigmoid(logits) >= 0.5).int().numpy()
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary", zero_division=0)
    print(f"Precision {p:.3f} Recall {r:.3f} F1 {f1:.3f}")