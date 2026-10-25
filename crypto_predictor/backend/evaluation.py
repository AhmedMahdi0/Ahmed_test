"""
Honest evaluation for direction models.

* Walk-forward splits: train on the past, test on the next block, repeat.
  A purge gap of `horizon` rows keeps labels of training rows from
  overlapping the test period.
* Baselines (majority class, persistence, always-up) so a model's score can
  be judged against "no skill".
* A simple long/short backtest with trading fees.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss


def walk_forward_splits(n: int, n_splits: int, initial_frac: float, purge: int,
                        min_train: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits over rows [0, n). Returns (train_idx, test_idx) pairs."""
    initial = int(n * initial_frac)
    test_size = (n - initial) // n_splits
    if initial < min_train or test_size < 20:
        raise ValueError(f"Not enough labelled candles for walk-forward evaluation (have {n})")
    splits = []
    for k in range(n_splits):
        test_start = initial + k * test_size
        test_end = n if k == n_splits - 1 else test_start + test_size
        train_end = test_start - purge
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def split_train_val(train_idx: np.ndarray, val_frac: float, purge: int) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological fit/validation split of a training block (validation = most recent rows)."""
    n_val = int(len(train_idx) * val_frac)
    if n_val < 20:
        return train_idx, np.array([], dtype=int)
    val_idx = train_idx[-n_val:]
    fit_idx = train_idx[:max(0, len(train_idx) - n_val - purge)]
    return fit_idx, val_idx


def _safe(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) or np.isinf(value) else round(value, 4)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, classes: List[str],
                           fwd_return: np.ndarray, confidence_levels=(0.5, 0.6)) -> Dict:
    """Metrics for one model's out-of-sample predictions."""
    y_true = y_true.astype(int)
    pred = proba.argmax(axis=1)
    n_classes = len(classes)
    up, down = classes.index('up'), classes.index('down')

    directional = np.isin(pred, [up, down]) & (fwd_return != 0)
    hit = ((pred == up) & (fwd_return > 0)) | ((pred == down) & (fwd_return < 0))
    clipped = np.clip(proba, 1e-6, 1)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)

    metrics = {
        'n': int(len(y_true)),
        'accuracy': _safe(accuracy_score(y_true, pred)),
        'balanced_accuracy': _safe(balanced_accuracy_score(y_true, pred)),
        'f1_macro': _safe(f1_score(y_true, pred, average='macro', labels=list(range(n_classes)), zero_division=0)),
        'log_loss': _safe(log_loss(y_true, clipped, labels=list(range(n_classes)))),
        # "When the model calls a direction, is the price actually moving that way?"
        'directional_accuracy': _safe(hit[directional].mean()) if directional.any() else None,
        'directional_coverage': _safe(directional.mean()),
        'precision_up': _safe(((pred == up) & (y_true == up)).sum() / max((pred == up).sum(), 1))
        if (pred == up).any() else None,
        'precision_down': _safe(((pred == down) & (y_true == down)).sum() / max((pred == down).sum(), 1))
        if (pred == down).any() else None,
        'confusion_matrix': confusion_matrix(y_true, pred, labels=list(range(n_classes))).tolist(),
        'confidence': {},
    }
    max_p = proba.max(axis=1)
    for level in confidence_levels:
        mask = max_p >= level
        metrics['confidence'][f'{level:.2f}'] = {
            'coverage': _safe(mask.mean()),
            'accuracy': _safe((pred[mask] == y_true[mask]).mean()) if mask.any() else None,
        }
    return metrics


def baseline_probas(y_train: np.ndarray, persistence: np.ndarray, n: int, classes: List[str]) -> Dict[str, np.ndarray]:
    """One-hot 'predictions' of naive strategies on the test block."""
    n_classes = len(classes)
    majority = int(np.bincount(y_train.astype(int), minlength=n_classes).argmax())
    out = {
        'baseline_majority': np.eye(n_classes)[np.full(n, majority)],
        'baseline_persistence': np.eye(n_classes)[persistence.astype(int)],
        'baseline_always_up': np.eye(n_classes)[np.full(n, classes.index('up'))],
    }
    return out


def positions_from_proba(proba: np.ndarray, classes: List[str], min_confidence: float, allow_short: bool) -> np.ndarray:
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1) >= min_confidence
    pos = np.zeros(len(pred))
    pos[(pred == classes.index('up')) & conf] = 1.0
    if allow_short:
        pos[(pred == classes.index('down')) & conf] = -1.0
    return pos


def backtest(proba: np.ndarray, next_return: np.ndarray, classes: List[str], bars_per_year: float,
             fee_bps: float, min_confidence: float, allow_short: bool, max_points: int = 400) -> Dict:
    """
    Trade each signal for one candle: long on 'up', short on 'down' (if allowed),
    flat otherwise. Fees are charged on every change of position.
    """
    pos = positions_from_proba(proba, classes, min_confidence, allow_short)
    fee = fee_bps / 10000.0
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * next_return - turnover * fee
    equity = np.cumprod(1 + strat)
    hold = np.cumprod(1 + next_return)

    peak = np.maximum.accumulate(equity)
    in_market = pos != 0
    std = strat.std()
    step = max(1, len(equity) // max_points)
    return {
        'total_return': _safe(equity[-1] - 1),
        'buy_hold_return': _safe(hold[-1] - 1),
        'sharpe': _safe(strat.mean() / std * np.sqrt(bars_per_year)) if std > 0 else None,
        'max_drawdown': _safe((equity / peak - 1).min()),
        'trades': int((turnover > 0).sum()),
        'exposure': _safe(in_market.mean()),
        'win_rate': _safe((strat[in_market] > 0).mean()) if in_market.any() else None,
        'fee_bps': fee_bps,
        'equity_curve': {
            'index': list(range(0, len(equity), step)),
            'strategy': [round(float(v), 5) for v in equity[::step]],
            'buy_hold': [round(float(v), 5) for v in hold[::step]],
        },
    }
