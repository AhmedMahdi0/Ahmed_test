"""
A/B test a feature group: run the same walk-forward analysis with and without
support/resistance features on identical data and compare out-of-sample results.

    cd crypto_predictor
    python -m experiments.compare_features --symbol BTC --timeframes 1h 4h 1d
    python -m experiments.compare_features --symbol ETH --models gbm logreg gru --folds 5
    python -m experiments.compare_features --csv my_candles.csv --csv-timeframe 1h --timeframes 1h 4h

Differences are paired candle-by-candle (both variants predict the same
candles), with 95% confidence intervals from a block bootstrap, since
neighbouring candles are not independent. Trained models are written to a
temporary directory, so the app's saved models are left untouched.
"""

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import get_config, get_timeframe, DEFAULT_MODELS
from backend.data_fetcher import CryptoDataFetcher
from backend.prediction_service import PredictionService

VARIANTS = {'baseline': False, 'support_resistance': True}


class CsvFetcher:
    """Serves candles from a CSV (columns: time index, open, high, low, close, volume)."""

    def __init__(self, path: str, timeframe: str):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index.name = 'time'
        for col in ('quote_volume', 'trades', 'taker_buy_base'):
            if col not in df:
                df[col] = df['close'] * df['volume'] if col == 'quote_volume' else np.nan
        self.df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base']].astype(float)
        self.timeframe = timeframe

    def fetch_ohlcv(self, symbol, timeframe, history_days=None):
        src, dst = get_timeframe(self.timeframe), get_timeframe(timeframe)
        if dst.minutes < src.minutes:
            raise ValueError(f"Cannot build {timeframe} candles from {self.timeframe} data")
        df = self.df
        if dst.minutes > src.minutes:
            g = df.resample(f'{dst.minutes}min', origin='epoch', label='left', closed='left')
            df = g.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                        'quote_volume': 'sum', 'trades': 'sum', 'taker_buy_base': 'sum'}).dropna(subset=['close'])
            df['taker_buy_base'] = np.nan
        df = df.copy()
        df.attrs['source'] = 'csv'
        return df

    def fetch_macro(self):
        return pd.DataFrame()


def block_bootstrap_ci(diff: np.ndarray, block: int, n_boot: int = 1000, seed: int = 0):
    """95% CI and two-sided p-value for the mean of a (serially correlated) difference series."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return float(lo), float(hi), float(p)


def paired_comparison(a: pd.DataFrame, b: pd.DataFrame, models, classes, block: int) -> dict:
    joined = a.add_suffix('_a').join(b.add_suffix('_b'), how='inner')
    actual = joined['actual_a']
    out = {'paired_candles': int(len(joined))}
    for name in models + ['ensemble']:
        correct_a = (joined[f'{name}_pred_a'] == actual).to_numpy(float)
        correct_b = (joined[f'{name}_pred_b'] == actual).to_numpy(float)
        p_a = np.array([joined[f'{name}_p_{c}_a'].to_numpy() for c in classes]).T
        p_b = np.array([joined[f'{name}_p_{c}_b'].to_numpy() for c in classes]).T
        cls = actual.map({c: i for i, c in enumerate(classes)}).to_numpy()
        ll_a = -np.log(np.clip(p_a[np.arange(len(cls)), cls], 1e-6, 1))
        ll_b = -np.log(np.clip(p_b[np.arange(len(cls)), cls], 1e-6, 1))
        acc_lo, acc_hi, acc_p = block_bootstrap_ci(correct_b - correct_a, block)
        ll_lo, ll_hi, ll_p = block_bootstrap_ci(ll_b - ll_a, block)
        out[name] = {
            'accuracy_baseline': round(correct_a.mean(), 4),
            'accuracy_sr': round(correct_b.mean(), 4),
            'accuracy_diff': round(correct_b.mean() - correct_a.mean(), 4),
            'accuracy_diff_ci95': [round(acc_lo, 4), round(acc_hi, 4)],
            'accuracy_p_value': round(acc_p, 3),
            'log_loss_diff': round(ll_b.mean() - ll_a.mean(), 4),   # negative = better
            'log_loss_diff_ci95': [round(ll_lo, 4), round(ll_hi, 4)],
        }
    return out


def verdict(cmp: dict) -> str:
    e = cmp['ensemble']
    lo, hi = e['accuracy_diff_ci95']
    ll_lo, ll_hi = e['log_loss_diff_ci95']
    if lo > 0 and ll_hi <= 0.002:
        return 'IMPROVED'
    if hi < 0 or ll_lo > 0:
        return 'WORSE'
    return 'NO SIGNIFICANT DIFFERENCE'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbol', default='BTC')
    parser.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--folds', type=int, help='walk-forward folds (default: current setting)')
    parser.add_argument('--label-mode', choices=['ternary', 'binary'])
    parser.add_argument('--csv', help='use candles from a CSV instead of downloading')
    parser.add_argument('--csv-timeframe', default='1h', help='timeframe of the CSV candles')
    parser.add_argument('--out', help='write the full comparison as JSON')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cfg = get_config().app_config
    saved = asdict(cfg)
    cfg.output_dir = tempfile.mkdtemp(prefix='sr_experiment_')
    if args.folds:
        cfg.walk_forward_splits = args.folds
    if args.label_mode:
        cfg.label_mode = args.label_mode
    if args.csv:
        fetcher = CsvFetcher(args.csv, args.csv_timeframe)
        cfg.use_btc_context = cfg.use_macro = False
    else:
        fetcher = CryptoDataFetcher()  # one instance: both variants see the same cached candles

    report = {'symbol': args.symbol, 'models': args.models, 'label_mode': cfg.label_mode,
              'source': args.csv or 'exchange', 'timeframes': {}}
    try:
        for tf in args.timeframes:
            runs = {}
            for variant, flag in VARIANTS.items():
                cfg.use_support_resistance = flag
                service = PredictionService(fetcher=fetcher)
                print(f"[{tf}] running {variant}...", file=sys.stderr)
                result = service.analyze(args.symbol, [tf], args.models)
                if tf not in result['timeframes']:
                    raise RuntimeError(f"{tf} failed: {result['errors'].get(tf)}")
                runs[variant] = (result['timeframes'][tf], service.oos[(result['symbol'], tf)])

            base_res, base_oos = runs['baseline']
            sr_res, sr_oos = runs['support_resistance']
            block = max(5, min(48, len(base_oos) // 50))
            cmp = paired_comparison(base_oos, sr_oos, args.models, base_res['label']['classes'], block)
            cmp['verdict'] = verdict(cmp)
            cmp['backtest_return'] = {'baseline': base_res['backtest']['total_return'],
                                      'support_resistance': sr_res['backtest']['total_return']}
            cmp['directional_accuracy'] = {
                name: {'baseline': base_res['metrics'][name]['directional_accuracy'],
                       'support_resistance': sr_res['metrics'][name]['directional_accuracy']}
                for name in args.models + ['ensemble']}
            cmp['sr_features_in_top10'] = [f['feature'] for f in sr_res['feature_importance'][:10]
                                           if f['feature'].startswith('sr_')]
            report['timeframes'][tf] = cmp
            print_table(tf, cmp, args.models)
    finally:
        cfg.update({k: v for k, v in saved.items()})

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nSaved {args.out}", file=sys.stderr)
    return report


def print_table(tf: str, cmp: dict, models):
    print(f"\n=== {tf}: {cmp['paired_candles']} out-of-sample candles -> {cmp['verdict']} ===")
    print(f"{'model':<12}{'acc base':>10}{'acc +S/R':>10}{'diff':>8}  {'95% CI':<18}{'p':>6}"
          f"{'dir base':>10}{'dir +S/R':>10}{'logloss Δ':>11}")
    for name in models + ['ensemble']:
        m, d = cmp[name], cmp['directional_accuracy'][name]
        ci = f"[{m['accuracy_diff_ci95'][0]:+.3f}, {m['accuracy_diff_ci95'][1]:+.3f}]"
        fmt = lambda v: '--' if v is None else f"{v:.3f}"
        print(f"{name:<12}{m['accuracy_baseline']:>10.3f}{m['accuracy_sr']:>10.3f}{m['accuracy_diff']:>+8.3f}  "
              f"{ci:<18}{m['accuracy_p_value']:>6.2f}{fmt(d['baseline']):>10}{fmt(d['support_resistance']):>10}"
              f"{m['log_loss_diff']:>+11.4f}")
    bt = cmp['backtest_return']
    print(f"backtest (ensemble): baseline {bt['baseline']:+.1%}  vs  +S/R {bt['support_resistance']:+.1%}")
    if cmp['sr_features_in_top10']:
        print(f"S/R features in GBM top-10: {', '.join(cmp['sr_features_in_top10'])}")


if __name__ == '__main__':
    main()
