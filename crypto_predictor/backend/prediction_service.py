"""
Prediction service for CryptoPredictor.

For each requested timeframe it:
  1. fetches candles (+ BTC and macro context),
  2. builds leakage-free features and direction labels,
  3. runs walk-forward evaluation of every selected model, the ensemble and
     naive baselines, plus a backtest,
  4. refits the models on all labelled data and predicts the direction of
     the next candle (the "signal").
"""

import io
import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import get_config, get_timeframe, TIMEFRAMES, DEFAULT_MODELS
from .data_fetcher import CryptoDataFetcher
from .evaluation import (walk_forward_splits, split_train_val, classification_metrics,
                         baseline_probas, backtest)
from .feature_engineering import FeatureEngineer, Dataset
from .models import (create_model, save_model, load_model, available_models, ensemble_proba,
                     MODEL_INFO, GBMModel)

logger = logging.getLogger(__name__)

HISTORY_POINTS = 300
MIN_LABELLED_ROWS = 400


class AnalysisCancelled(Exception):
    """Raised when the caller cancels a running analysis."""


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat() + 'Z'


def _round(value, digits=4):
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, digits)


class PredictionService:
    """Runs analyses and keeps the latest trained models per (symbol, timeframe)."""

    def __init__(self, fetcher: Optional[CryptoDataFetcher] = None):
        self.config = get_config()
        self.fetcher = fetcher or CryptoDataFetcher()
        self.features = FeatureEngineer()
        self._lock = threading.Lock()
        self.results: Dict[tuple, Dict] = {}      # (symbol, tf) -> analysis result (JSON-ready)
        self.oos: Dict[tuple, pd.DataFrame] = {}  # (symbol, tf) -> out-of-sample predictions
        self.bundles: Dict[tuple, Dict] = {}      # (symbol, tf) -> {'models': {...}, 'meta': {...}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, timeframes: Optional[List[str]] = None, models: Optional[List[str]] = None,
                progress_callback: Optional[Callable] = None, cancel_event: Optional[threading.Event] = None) -> Dict:
        symbol = self.config.get_crypto(symbol).symbol
        timeframes = timeframes or list(TIMEFRAMES)
        for tf in timeframes:
            get_timeframe(tf)
        models = self._validate_models(models or DEFAULT_MODELS)

        def report(fraction: float, message: str):
            if progress_callback:
                progress_callback(round(min(fraction, 0.99) * 100, 1), message)

        def cancel_check():
            if cancel_event is not None and cancel_event.is_set():
                raise AnalysisCancelled("Analysis cancelled")

        started = time.time()
        out = {'symbol': symbol, 'generated_at': _iso(datetime.now(timezone.utc).replace(tzinfo=None)),
               'models': models, 'timeframes': {}, 'errors': {}}
        for i, tf in enumerate(timeframes):
            share = 1.0 / len(timeframes)
            sub = lambda f, m, i=i, tf=tf: report((i + f) * share, f"[{tf}] {m}")
            try:
                out['timeframes'][tf] = self._analyze_timeframe(symbol, tf, models, sub, cancel_check)
            except AnalysisCancelled:
                raise
            except Exception as e:
                logger.exception("Analysis failed for %s %s", symbol, tf)
                out['errors'][tf] = str(e)
        if not out['timeframes']:
            raise ValueError(f"Analysis failed for every timeframe: {out['errors']}")
        out['elapsed_seconds'] = round(time.time() - started, 1)
        out['settings'] = self._settings_snapshot()
        return out

    def latest_signal(self, symbol: str, timeframes: Optional[List[str]] = None) -> Dict:
        """Refresh candles and re-predict the next candle with already-trained models (no retraining)."""
        symbol = self.config.get_crypto(symbol).symbol
        timeframes = timeframes or list(TIMEFRAMES)
        out = {'symbol': symbol, 'generated_at': _iso(datetime.now(timezone.utc).replace(tzinfo=None)),
               'signals': {}, 'errors': {}}
        for tf in timeframes:
            try:
                bundle = self._get_bundle(symbol, tf)
                ds = self._dataset(symbol, tf)
                if ds.classes != bundle['meta']['classes'] or ds.horizon != bundle['meta']['horizon']:
                    raise ValueError("Label settings changed since training. Re-run the analysis.")
                missing = [c for c in bundle['meta']['feature_columns'] if c not in ds.X.columns]
                if missing:
                    raise ValueError(f"Features changed since training ({missing[:3]}...). Re-run the analysis.")
                X = ds.X[bundle['meta']['feature_columns']].to_numpy(dtype=np.float32)
                out['signals'][tf] = self._signal(ds, X, bundle['models'], bundle['meta'].get('reliability'))
            except Exception as e:
                out['errors'][tf] = str(e)
        return out

    def get_result(self, symbol: str, timeframe: str) -> Optional[Dict]:
        with self._lock:
            return self.results.get((symbol.upper(), timeframe))

    def export_csv(self, symbol: str, timeframe: str) -> str:
        with self._lock:
            df = self.oos.get((symbol.upper(), timeframe))
        if df is None:
            raise ValueError(f"No analysis for {symbol} {timeframe} yet")
        buf = io.StringIO()
        df.to_csv(buf, index_label='time')
        return buf.getvalue()

    def get_available_cryptos(self) -> List[Dict]:
        return [{'symbol': c.symbol, 'name': c.name, 'enabled': c.enabled} for c in self.config.cryptocurrencies]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_models(models: List[str]) -> List[str]:
        avail = available_models()
        unknown = [m for m in models if m not in MODEL_INFO]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}. Available: {list(MODEL_INFO)}")
        usable = [m for m in models if avail[m]]
        if not usable:
            raise ValueError("None of the selected models is available (PyTorch missing?)")
        return usable

    def _dataset(self, symbol: str, tf: str) -> Dataset:
        cfg = self.config.app_config
        ohlcv = self.fetcher.fetch_ohlcv(symbol, tf)
        btc = None
        if cfg.use_btc_context and symbol != 'BTC':
            try:
                btc = self.fetcher.fetch_ohlcv('BTC', tf)
            except Exception as e:
                logger.warning("BTC context unavailable: %s", e)
        macro = None
        if cfg.use_macro:
            try:
                macro = self.fetcher.fetch_macro()
            except Exception as e:
                logger.warning("Macro context unavailable: %s", e)
        ds = self.features.make_dataset(ohlcv, tf, btc=btc, macro=macro)
        ds.X.attrs['source'] = ohlcv.attrs.get('source')
        return ds

    def _analyze_timeframe(self, symbol: str, tf: str, model_names: List[str], report, cancel_check) -> Dict:
        cfg = self.config.app_config
        spec = get_timeframe(tf)
        report(0.02, 'Fetching candles...')
        ds = self._dataset(symbol, tf)
        cancel_check()

        labelled = np.flatnonzero(ds.labeled_mask)
        if len(labelled) < MIN_LABELLED_ROWS:
            raise ValueError(f"Only {len(labelled)} labelled {tf} candles for {symbol}; need {MIN_LABELLED_ROWS}")
        # Labelled rows are a contiguous prefix (only the last `horizon` rows lack a label).
        n = labelled[-1] + 1
        X = ds.X.to_numpy(dtype=np.float32)
        y = ds.y.to_numpy(dtype=float)
        fwd = ds.fwd_return.to_numpy()
        next_ret = np.nan_to_num(ds.next_return.to_numpy())
        persistence = ds.persistence.to_numpy()
        n_classes = len(ds.classes)
        purge = ds.horizon

        splits = walk_forward_splits(n, cfg.walk_forward_splits, cfg.initial_train_frac, purge)
        test_rows = np.concatenate([test for _, test in splits])
        oos = {m: np.zeros((len(test_rows), n_classes)) for m in model_names}
        base = {}
        total_steps = len(splits) * len(model_names) + len(model_names)
        step = 0

        offset = 0
        for k, (train_idx, test_idx) in enumerate(splits):
            fit_idx, val_idx = split_train_val(train_idx, cfg.val_frac, purge)
            for name in model_names:
                cancel_check()
                report(0.05 + 0.8 * step / total_steps, f"Walk-forward fold {k + 1}/{len(splits)}: {name}")
                model = create_model(name, n_classes, tf).fit(X, y, fit_idx, val_idx, cancel_check=cancel_check)
                oos[name][offset:offset + len(test_idx)] = model.predict_proba(X, test_idx)
                step += 1
            p = np.nan_to_num(persistence[test_idx], nan=1 if n_classes == 3 else 0)
            for bname, bproba in baseline_probas(y[train_idx], p, len(test_idx), ds.classes).items():
                base.setdefault(bname, []).append(bproba)
            offset += len(test_idx)

        oos['ensemble'] = ensemble_proba([oos[m] for m in model_names])
        base = {k: np.concatenate(v) for k, v in base.items()}
        y_test, fwd_test, next_test = y[test_rows], fwd[test_rows], next_ret[test_rows]

        metrics = {name: classification_metrics(y_test, proba, ds.classes, fwd_test)
                   for name, proba in {**oos, **base}.items()}
        bt_kwargs = dict(classes=ds.classes, bars_per_year=spec.bars_per_year, fee_bps=cfg.fee_bps,
                         min_confidence=cfg.min_confidence, allow_short=cfg.allow_short)
        test_times = ds.X.index[test_rows]
        bt = backtest(oos['ensemble'], next_test, **bt_kwargs)
        bt['equity_curve']['time'] = [_iso(test_times[i]) for i in bt['equity_curve'].pop('index')]
        for name in model_names:
            metrics[name]['backtest_return'] = backtest(oos[name], next_test, **bt_kwargs)['total_return']

        reliability = self._reliability(oos['ensemble'], y_test, fwd_test, ds.classes)

        # Final models on all labelled rows -> live signal for the latest closed candle.
        fit_idx, val_idx = split_train_val(np.arange(n), cfg.val_frac, purge)
        final = {}
        for name in model_names:
            cancel_check()
            report(0.05 + 0.8 * step / total_steps, f"Training final {name} model")
            final[name] = create_model(name, n_classes, tf).fit(X, y, fit_idx, val_idx, cancel_check=cancel_check)
            step += 1
        report(0.95, 'Generating signal...')
        signal = self._signal(ds, X, final, reliability)

        meta = {
            'symbol': symbol, 'timeframe': tf, 'classes': ds.classes, 'feature_columns': list(ds.X.columns),
            'trained_at': _iso(datetime.now(timezone.utc).replace(tzinfo=None)), 'models': model_names,
            'threshold': ds.threshold, 'horizon': ds.horizon, 'reliability': reliability,
            'settings': self._settings_snapshot(),
        }
        self._save_bundle(symbol, tf, final, meta)

        oos_df = pd.DataFrame(index=test_times)
        oos_df['close'] = ds.close.to_numpy()[test_rows]
        oos_df['fwd_log_return'] = fwd_test
        oos_df['actual'] = [ds.classes[int(v)] for v in y_test]
        for name in model_names + ['ensemble']:
            oos_df[f'{name}_pred'] = [ds.classes[i] for i in oos[name].argmax(axis=1)]
            for c, cname in enumerate(ds.classes):
                oos_df[f'{name}_p_{cname}'] = oos[name][:, c].round(4)

        result = {
            'timeframe': tf,
            'data': {
                'source': ds.X.attrs.get('source'), 'candles': int(len(ds.X)), 'labelled': int(n),
                'features': int(X.shape[1]), 'start': _iso(ds.X.index[0]), 'end': _iso(ds.X.index[-1]),
                'test_start': _iso(test_times[0]), 'test_candles': int(len(test_rows)),
                'class_balance': {c: _round((y_test == i).mean()) for i, c in enumerate(ds.classes)},
            },
            'label': {'mode': cfg.label_mode, 'classes': ds.classes, 'threshold': ds.threshold,
                      'horizon': ds.horizon},
            'signal': signal,
            'metrics': metrics,
            'backtest': bt,
            'reliability': reliability,
            'history': self._history(oos_df, ds.classes),
            'feature_importance': self._importance(final.get('gbm'), list(ds.X.columns)),
            'walk_forward': {'splits': len(splits), 'purge': purge},
        }
        with self._lock:
            self.results[(symbol, tf)] = result
            self.oos[(symbol, tf)] = oos_df
        return result

    def _signal(self, ds: Dataset, X: np.ndarray, models: Dict, reliability: Optional[Dict]) -> Dict:
        spec = get_timeframe(ds.timeframe)
        last = np.array([len(X) - 1])
        per_model = {name: m.predict_proba(X, last)[0] for name, m in models.items()}
        proba = np.mean(list(per_model.values()), axis=0)
        pred = int(proba.argmax())
        direction = ds.classes[pred]
        votes = [ds.classes[int(p.argmax())] for p in per_model.values()]
        agreement = votes.count(direction) / len(votes)
        confidence = float(proba.max())
        level = self._confidence_level(confidence, agreement, len(ds.classes))

        candle_open = ds.X.index[-1]
        candle_close = candle_open + pd.Timedelta(minutes=spec.minutes)
        target = candle_close + pd.Timedelta(minutes=spec.minutes * ds.horizon)
        signal = {
            'direction': direction,
            'probabilities': {c: _round(proba[i]) for i, c in enumerate(ds.classes)},
            'confidence': _round(confidence),
            'confidence_level': level,
            'agreement': _round(agreement),
            'models': {name: {'direction': ds.classes[int(p.argmax())],
                              'probabilities': {c: _round(p[i]) for i, c in enumerate(ds.classes)}}
                       for name, p in per_model.items()},
            'price': _round(ds.close.iloc[-1], 8),
            'candle_open': _iso(candle_open),
            'based_on_close': _iso(candle_close),
            'target_time': _iso(target),
            'threshold': ds.threshold,
        }
        if reliability:
            bucket = reliability.get(direction, {}).get(level)
            signal['historical'] = bucket
        return signal

    @staticmethod
    def _confidence_level(confidence: float, agreement: float, n_classes: int) -> str:
        chance = 1.0 / n_classes
        if confidence >= chance + 0.22 and agreement >= 0.99:
            return 'high'
        if confidence >= chance + 0.10 and agreement >= 0.5:
            return 'medium'
        return 'low'

    def _reliability(self, proba: np.ndarray, y: np.ndarray, fwd: np.ndarray, classes: List[str]) -> Dict:
        """How often past out-of-sample signals of each direction/confidence level were right."""
        pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        # Agreement is not tracked out-of-sample per row, so levels here use confidence alone.
        levels = np.array([self._confidence_level(c, 1.0, len(classes)) for c in conf])
        out = {}
        for i, cname in enumerate(classes):
            out[cname] = {}
            for level in ('low', 'medium', 'high'):
                mask = (pred == i) & (levels == level)
                if not mask.any():
                    out[cname][level] = {'n': 0, 'hit_rate': None, 'moved_that_way': None}
                    continue
                moved = None
                if cname == 'up':
                    moved = (fwd[mask] > 0).mean()
                elif cname == 'down':
                    moved = (fwd[mask] < 0).mean()
                out[cname][level] = {'n': int(mask.sum()), 'hit_rate': _round((y[mask] == i).mean()),
                                     'moved_that_way': _round(moved)}
        return out

    @staticmethod
    def _history(oos_df: pd.DataFrame, classes: List[str]) -> Dict:
        tail = oos_df.tail(HISTORY_POINTS)
        return {
            'time': [_iso(t) for t in tail.index],
            'close': [_round(v, 8) for v in tail['close']],
            'actual': tail['actual'].tolist(),
            'predicted': tail['ensemble_pred'].tolist(),
            'confidence': [_round(v) for v in tail[[f'ensemble_p_{c}' for c in classes]].max(axis=1)],
        }

    @staticmethod
    def _importance(model, columns: List[str], top: int = 15) -> List[Dict]:
        if not isinstance(model, GBMModel):
            return []
        imp = model.feature_importance()
        if imp is None or imp.sum() == 0:
            return []
        imp = imp / imp.sum()
        order = np.argsort(imp)[::-1][:top]
        return [{'feature': columns[i], 'importance': _round(imp[i])} for i in order]

    def _settings_snapshot(self) -> Dict:
        return asdict(self.config.app_config)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _bundle_dir(self, symbol: str, tf: str) -> Path:
        return Path(self.config.get_output_path('models', f'{symbol}_{tf}', 'meta.json')).parent

    def _save_bundle(self, symbol: str, tf: str, models: Dict, meta: Dict) -> None:
        with self._lock:
            self.bundles[(symbol, tf)] = {'models': models, 'meta': meta}
        try:
            directory = self._bundle_dir(symbol, tf)
            for model in models.values():
                save_model(model, directory)
            (directory / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
        except Exception as e:
            logger.warning("Could not save models for %s %s: %s", symbol, tf, e)

    def _get_bundle(self, symbol: str, tf: str) -> Dict:
        with self._lock:
            bundle = self.bundles.get((symbol, tf))
        if bundle is not None:
            return bundle
        directory = self._bundle_dir(symbol, tf)
        meta_path = directory / 'meta.json'
        if not meta_path.exists():
            raise ValueError(f"No trained models for {symbol} {tf}. Run an analysis first.")
        meta = json.loads(meta_path.read_text())
        models = {name: load_model(directory, name) for name in meta['models']}
        bundle = {'models': models, 'meta': meta}
        with self._lock:
            self.bundles[(symbol, tf)] = bundle
        return bundle
