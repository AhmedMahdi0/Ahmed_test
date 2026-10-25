"""
Direction classifiers for CryptoPredictor.

Every model shares one interface built around the *full* feature matrix and
row indices, so sequence models can look back over the candles that precede
a row even when those candles belong to an earlier split:

    model.fit(X, y, fit_idx, val_idx)      # X: (n_rows, n_features), y: class ids
    model.predict_proba(X, idx)            # -> (len(idx), n_classes)

Tabular models: gradient boosting ('gbm') and logistic regression ('logreg').
Sequence models (need PyTorch): 'gru', 'lstm', 'transformer'.
"""

import copy
import inspect
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import get_config

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    lgb = None
    LIGHTGBM_AVAILABLE = False


MODEL_INFO = {
    'gbm': {'label': 'Gradient Boosting', 'description': 'LightGBM on tabular features - strong baseline'},
    'logreg': {'label': 'Logistic Regression', 'description': 'Linear baseline, fast and robust'},
    'gru': {'label': 'GRU', 'description': 'Recurrent net over a window of past candles'},
    'lstm': {'label': 'LSTM', 'description': 'Recurrent net with long-term memory'},
    'transformer': {'label': 'Transformer', 'description': 'Self-attention over past candles'},
}
SEQUENCE_MODELS = ('gru', 'lstm', 'transformer')


def available_models() -> Dict[str, bool]:
    return {name: (TORCH_AVAILABLE if name in SEQUENCE_MODELS else True) for name in MODEL_INFO}


def balanced_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    weights = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return weights


def _full_proba(proba: np.ndarray, model_classes, n_classes: int) -> np.ndarray:
    """Expand predict_proba output to all classes (a class may be missing from a training fold)."""
    out = np.zeros((proba.shape[0], n_classes))
    out[:, np.asarray(model_classes, dtype=int)] = proba
    return out


class DirectionModel:
    """Base class."""
    name = 'base'

    def __init__(self, n_classes: int, timeframe: str):
        self.n_classes = n_classes
        self.timeframe = timeframe
        self.config = get_config().app_config

    def fit(self, X: np.ndarray, y: np.ndarray, fit_idx: np.ndarray, val_idx: Optional[np.ndarray] = None,
            cancel_check=None) -> 'DirectionModel':
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def min_history(self) -> int:
        """Rows of history required before the first predictable row."""
        return 0

    def save(self, path: Path) -> None:
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> 'DirectionModel':
        with open(path, 'rb') as fh:
            return pickle.load(fh)


# ============================================
# Tabular models
# ============================================

class GBMModel(DirectionModel):
    name = 'gbm'

    def fit(self, X, y, fit_idx, val_idx=None, cancel_check=None):
        cfg = self.config
        y_fit = y[fit_idx].astype(int)
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=cfg.gbm_estimators,
                learning_rate=cfg.gbm_learning_rate,
                num_leaves=cfg.gbm_num_leaves,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                class_weight='balanced',
                random_state=cfg.seed,
                verbose=-1,
            )
            kwargs = {}
            if val_idx is not None and len(val_idx) > 0:
                if 'eval_X' in inspect.signature(self.model.fit).parameters:   # lightgbm >= 4.6
                    kwargs.update(eval_X=(X[val_idx],), eval_y=(y[val_idx].astype(int),))
                else:
                    kwargs['eval_set'] = [(X[val_idx], y[val_idx].astype(int))]
                kwargs['callbacks'] = [lgb.early_stopping(50, verbose=False)]
            self.model.fit(X[fit_idx], y_fit, **kwargs)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=min(cfg.gbm_estimators, 300),
                learning_rate=cfg.gbm_learning_rate * 2,
                max_leaf_nodes=cfg.gbm_num_leaves,
                min_samples_leaf=50,
                l2_regularization=1.0,
                class_weight='balanced',
                early_stopping=False,  # its internal split is random, which would leak time order
                random_state=cfg.seed,
            )
            self.model.fit(X[fit_idx], y_fit)
        return self

    def predict_proba(self, X, idx):
        return _full_proba(self.model.predict_proba(X[idx]), self.model.classes_, self.n_classes)

    def feature_importance(self) -> Optional[np.ndarray]:
        if LIGHTGBM_AVAILABLE and hasattr(self.model, 'booster_'):
            return self.model.booster_.feature_importance(importance_type='gain')
        return None


class LogRegModel(DirectionModel):
    name = 'logreg'

    def fit(self, X, y, fit_idx, val_idx=None, cancel_check=None):
        self.scaler = StandardScaler().fit(X[fit_idx])
        self.model = LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced')
        self.model.fit(self.scaler.transform(X[fit_idx]), y[fit_idx].astype(int))
        return self

    def predict_proba(self, X, idx):
        proba = self.model.predict_proba(self.scaler.transform(X[idx]))
        return _full_proba(proba, self.model.classes_, self.n_classes)


# ============================================
# Sequence models (PyTorch)
# ============================================

if TORCH_AVAILABLE:

    class _RNNNet(nn.Module):
        def __init__(self, cell: str, n_features: int, hidden: int, layers: int, dropout: float, n_classes: int):
            super().__init__()
            rnn_cls = nn.GRU if cell == 'gru' else nn.LSTM
            self.rnn = rnn_cls(n_features, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

        def forward(self, x):                      # x: (batch, window, features)
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    class _TransformerNet(nn.Module):
        def __init__(self, n_features: int, hidden: int, layers: int, dropout: float, n_classes: int, window: int):
            super().__init__()
            heads = 4 if hidden % 4 == 0 else 1
            self.embed = nn.Linear(n_features, hidden)
            self.pos = nn.Parameter(torch.zeros(1, window, hidden))
            layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 2,
                                               dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, layers))
            self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

        def forward(self, x):
            h = self.encoder(self.embed(x) + self.pos[:, -x.shape[1]:, :])
            return self.head(torch.cat([h[:, -1, :], h.mean(dim=1)], dim=1))


class SequenceModel(DirectionModel):
    """Classifier over a window of the most recent `window` candles."""

    def __init__(self, name: str, n_classes: int, timeframe: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; sequence models are unavailable")
        super().__init__(n_classes, timeframe)
        self.name = name
        self.window = self.config.window_for(timeframe)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def min_history(self) -> int:
        return self.window - 1

    def _build(self, n_features: int):
        cfg = self.config
        if self.name == 'transformer':
            return _TransformerNet(n_features, cfg.hidden_dim, max(2, cfg.num_layers), cfg.dropout,
                                   self.n_classes, self.window)
        return _RNNNet(self.name, n_features, cfg.hidden_dim, cfg.num_layers, cfg.dropout, self.n_classes)

    def _windows(self, Xt, idx: np.ndarray):
        offsets = torch.arange(-self.window + 1, 1, device=Xt.device)
        rows = torch.as_tensor(idx, device=Xt.device).unsqueeze(1) + offsets
        return Xt[rows]                                           # (batch, window, features)

    def _usable(self, idx: np.ndarray) -> np.ndarray:
        return idx[idx >= self.window - 1]

    def fit(self, X, y, fit_idx, val_idx=None, cancel_check=None):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        fit_idx = self._usable(fit_idx)
        val_idx = self._usable(val_idx) if val_idx is not None else np.array([], dtype=int)
        if len(fit_idx) < 50:
            raise ValueError(f"Not enough rows to train {self.name} (have {len(fit_idx)})")

        # Scale with statistics from rows the model is allowed to see.
        visible_end = fit_idx.max() + 1
        self.scaler = StandardScaler().fit(X[:visible_end])
        Xt = torch.as_tensor(np.clip(self.scaler.transform(X), -8, 8), dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(np.nan_to_num(y, nan=0).astype(np.int64), device=self.device)

        self.net = self._build(X.shape[1]).to(self.device)
        weights = torch.as_tensor(balanced_class_weights(y[fit_idx], self.n_classes),
                                  dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        best_loss, best_state, bad_epochs = float('inf'), None, 0
        for epoch in range(cfg.epochs):
            if cancel_check:
                cancel_check()
            self.net.train()
            order = rng.permutation(fit_idx)
            for start in range(0, len(order), cfg.batch_size):
                batch = order[start:start + cfg.batch_size]
                optimizer.zero_grad()
                loss = criterion(self.net(self._windows(Xt, batch)), yt[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

            if len(val_idx) == 0:
                continue
            val_loss = self._loss(Xt, yt, val_idx, criterion)
            if val_loss < best_loss - 1e-4:
                best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info("%s early stop at epoch %d (val loss %.4f)", self.name, epoch + 1, best_loss)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def _loss(self, Xt, yt, idx, criterion) -> float:
        self.net.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(idx), 1024):
                batch = idx[start:start + 1024]
                total += criterion(self.net(self._windows(Xt, batch)), yt[batch]).item() * len(batch)
                count += len(batch)
        return total / max(count, 1)

    def predict_proba(self, X, idx):
        idx = np.asarray(idx)
        out = np.full((len(idx), self.n_classes), 1.0 / self.n_classes)
        ok = idx >= self.window - 1
        if not ok.any():
            return out
        Xt = torch.as_tensor(np.clip(self.scaler.transform(X), -8, 8), dtype=torch.float32, device=self.device)
        self.net.eval()
        preds = []
        with torch.no_grad():
            usable = idx[ok]
            for start in range(0, len(usable), 1024):
                logits = self.net(self._windows(Xt, usable[start:start + 1024]))
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        out[ok] = np.concatenate(preds)
        return out

    def save(self, path: Path) -> None:
        torch.save({
            'name': self.name, 'n_classes': self.n_classes, 'timeframe': self.timeframe,
            'window': self.window, 'n_features': int(self.scaler.n_features_in_),
            'hparams': {k: getattr(self.config, k) for k in ('hidden_dim', 'num_layers', 'dropout')},
            'scaler': self.scaler, 'state_dict': self.net.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(cls, path: Path) -> 'SequenceModel':
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(ckpt['name'], ckpt['n_classes'], ckpt['timeframe'])
        model.window = ckpt['window']
        model.scaler = ckpt['scaler']
        # Rebuild with the hyperparameters used at training time, not the current config.
        saved = copy.copy(model.config)
        for k, v in ckpt['hparams'].items():
            setattr(saved, k, v)
        model.config = saved
        model.net = model._build(ckpt['n_features']).to(model.device)
        model.net.load_state_dict(ckpt['state_dict'])
        model.net.eval()
        return model


# ============================================
# Factory / persistence
# ============================================

def create_model(name: str, n_classes: int, timeframe: str) -> DirectionModel:
    if name == 'gbm':
        return GBMModel(n_classes, timeframe)
    if name == 'logreg':
        return LogRegModel(n_classes, timeframe)
    if name in SEQUENCE_MODELS:
        return SequenceModel(name, n_classes, timeframe)
    raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_INFO)}")


def save_model(model: DirectionModel, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"{model.name}.pt" if isinstance(model, SequenceModel) else f"{model.name}.pkl")
    model.save(path)
    return path


def load_model(directory: Path, name: str) -> DirectionModel:
    if name in SEQUENCE_MODELS:
        return SequenceModel.load_checkpoint(directory / f"{name}.pt")
    return DirectionModel.load(directory / f"{name}.pkl")


def ensemble_proba(probas: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probas), axis=0)
