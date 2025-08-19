# ufeye/trainers.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ---------- utilities ----------
def bce_logits_loss(pos_weight: Optional[float] = None):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

@torch.no_grad()
def evaluate_auc(model, loader, device="cuda"):
    model.eval()
    ys, ps = [], []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x).view(-1)
        prob = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float('nan')

# ---------- supervised trainer ----------
class SupervisedTrainer:
    def __init__(self, model: nn.Module, lr=1e-4, weight_decay=1e-4, pos_weight: Optional[float]=None, device="cuda"):
        self.model = model.to(device)
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)
        self.loss_fn = bce_logits_loss(pos_weight)
        self.device = device

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs=20, early_stop=5):
        best_auc, best_state, no_improve = -1.0, None, 0
        for ep in range(1, epochs+1):
            self.model.train()
            for x,y in train_loader:
                x,y = x.to(self.device), y.to(self.device)
                logits = self.model(x).view(-1)
                loss = self.loss_fn(logits, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            val_auc = evaluate_auc(self.model, val_loader, self.device)
            if val_auc > best_auc:
                best_auc, best_state, no_improve = val_auc, {k:v.cpu() for k,v in self.model.state_dict().items()}, 0
            else:
                no_improve += 1
            if no_improve >= early_stop: break
        if best_state is not None:
            self.model.load_state_dict(best_state)

# ---------- autoencoder (unsupervised) ----------
class AETrainer:
    def __init__(self, ae: nn.Module, lr=1e-3, weight_decay=0.0, device="cuda"):
        self.ae = ae.to(device)
        self.optim = torch.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.threshold_: Optional[float] = None

    def fit_controls(self, control_loader: DataLoader, epochs=10):
        self.ae.train()
        for ep in range(epochs):
            for x,_ in control_loader:
                x = x.to(self.device)
                xhat, _ = self.ae(x)
                loss = self.loss_fn(xhat, x)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    @torch.no_grad()
    def calibrate_threshold(self, control_loader: DataLoader, quantile: float = 0.95):
        errs = []
        self.ae.eval()
        for x,_ in control_loader:
            x = x.to(self.device)
            xhat,_ = self.ae(x)
            err = torch.mean((xhat - x) ** 2, dim=list(range(1,x.ndim)))  # per-sample MSE
            errs.append(err.detach().cpu().numpy())
        errs = np.concatenate(errs)
        self.threshold_ = float(np.quantile(errs, quantile))

    @torch.no_grad()
    def predict_scores(self, loader: DataLoader):
        self.ae.eval()
        scores = []
        for x,_ in loader:
            x = x.to(self.device)
            xhat,_ = self.ae(x)
            err = torch.mean((xhat - x) ** 2, dim=list(range(1,x.ndim)))
            scores.append(err.detach().cpu().numpy())
        return np.concatenate(scores)

# ---------- PCA + (MLP / RF / XGB) ----------
class TorchMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x).view(-1)

def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)

class PCAPipeline:
    """
    Fits PCA on flattened images/volumes, then classifier:
      - 'mlp': Torch MLP (with dropout)
      - 'rf' : sklearn RandomForest
      - 'xgb': XGBoost (if available)
    """
    def __init__(self, n_components=100, clf="mlp", device="cuda"):
        self.ipca = IncrementalPCA(n_components=n_components, batch_size=256)
        self.clf_name = clf
        self.device = device
        self.mlp: Optional[TorchMLP] = None
        self.rf: Optional[RandomForestClassifier] = None
        self.xgb: Optional[XGBClassifier] = None

    def _collect_numpy(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for x,y in loader:
            if x.ndim == 5:  # [B,C,D,H,W]
                x = x.float()  # keep as is
            else:            # [B,C,H,W]
                x = x.float()
            x = x.view(x.size(0), -1).cpu().numpy()
            y = y.cpu().numpy()
            Xs.append(x); ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 20, lr: float = 1e-3):
        # Fit PCA incrementally
        for x,y in train_loader:
            Xb = x.view(x.size(0), -1).cpu().numpy()
            self.ipca.partial_fit(Xb)

        Xtr, ytr = self._collect_numpy(train_loader)
        Ztr = self.ipca.transform(Xtr)

        if self.clf_name == "mlp":
            self.mlp = TorchMLP(Ztr.shape[1]).to(self.device)
            opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()
            # simple split for val if provided
            if val_loader is not None:
                Xv, yv = self._collect_numpy(val_loader)
                Zv = self.ipca.transform(Xv)
                Xv = torch.from_numpy(Zv).float().to(self.device)
                yv = torch.from_numpy(yv).float().to(self.device)
            for ep in range(epochs):
                self.mlp.train()
                # mini-batch over numpy (simple)
                bs = 256
                idx = np.random.permutation(len(Ztr))
                for i in range(0, len(Ztr), bs):
                    j = idx[i:i+bs]
                    xb = torch.from_numpy(Ztr[j]).float().to(self.device)
                    yb = torch.from_numpy(ytr[j]).float().to(self.device)
                    logits = self.mlp(xb)
                    loss = loss_fn(logits, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
        elif self.clf_name == "rf":
            self.rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.rf.fit(Ztr, ytr)
        elif self.clf_name == "xgb":
            if not _HAS_XGB:
                raise RuntimeError("xgboost is not installed. Please `pip install xgboost` or use clf='rf'/'mlp'.")
            self.xgb = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", random_state=42
            )
            self.xgb.fit(Ztr, ytr)
        else:
            raise ValueError("clf must be 'mlp', 'rf', or 'xgb'")

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        X, _ = self._collect_numpy(loader)
        Z = self.ipca.transform(X)
        if self.clf_name == "mlp":
            zt = torch.from_numpy(Z).float().to(self.device)
            logits = self.mlp(zt)
            return torch.sigmoid(logits).cpu().numpy()
        elif self.clf_name == "rf":
            return self.rf.predict_proba(Z)[:,1]
        else:
            return self.xgb.predict_proba(Z)[:,1]
