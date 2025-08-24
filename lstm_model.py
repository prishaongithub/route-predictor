
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class LSTMRisk(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,T,F]
        out, _ = self.lstm(x)
        logits = self.head(out).squeeze(-1) # [B,T]
        return logits

def train_lstm(X_train, y_train, X_val, y_val, epochs=10, lr=1e-3, weight_decay=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = LSTMRisk(X_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X_train)
        loss = bce(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            preds = torch.sigmoid(val_logits).detach().cpu().numpy().ravel()
            true = y_val.detach().cpu().numpy().ravel()
            # avoid errors if only one class
            if len(np.unique(true)) == 1:
                auc = float("nan")
            else:
                auc = roc_auc_score(true, preds)
        print(f"Epoch {epoch+1}/{epochs} - loss {loss.item():.4f} - val AUC {auc:.4f}")
    return model

def infer_lstm(model, X):
    device = next(model.parameters()).device
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs  # [B,T]
