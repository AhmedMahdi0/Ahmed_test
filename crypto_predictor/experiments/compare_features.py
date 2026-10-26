"""
A/B test one change: run the same walk-forward analysis with the change off (A)
and on (B) on identical data and compare out-of-sample results.

Experiments:
    sr         support / resistance features
    htf        higher-timeframe context
    futures    perpetual futures data (funding, premium, open interest)
    drop_weak  leave weak moves (inside the neutral band) out of training

    cd crypto_predictor
    python -m experiments.compare_features --experiment futures --symbol BTC --timeframes 1h 4h 1d
    python -m experiments.compare_features --experiment htf --symbol ETH --models gbm logreg gru --folds 5
    python -m experiments.compare_features --experiment sr --csv candles.csv --csv-timeframe 1h --timeframes 1h 4h

Every run also checks the weak-signal filter: are the signals it shows more
accurate than the ones it hides?

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

EXPERIMENTS = {
    # name: (settings for A, settings for B, feature prefix shown in the importance check)
    'sr': ({'use_support_resistance': False}, {'use_support_resistance': True}, 'sr_'),
    'htf': ({'use_htf_context': False}, {'use_htf_context': True}, 'htf_'),
    'futures': ({'use_futures': False}, {'use_futures': True}, 'fut_'),
    'drop_weak': ({'label_mode': 'binary', 'drop_weak_moves': False},
                  {'label_mode': 'binary', 'drop_weak_moves': True}, None),
}


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
            'accuracy_a': round(correct_a.mean(), 4),
            'accuracy_b': round(correct_b.mean(), 4),
            'accuracy_diff': round(correct_b.mean() - correct_a.mean(), 4),
            'accuracy_diff_ci95': [round(acc_lo, 4), round(acc_hi, 4)],
            'accuracy_p_value': round(acc_p, 3),
            'log_loss_diff': round(ll_b.mean() - ll_a.mean(), 4),   # negative = better
            'log_loss_diff_ci95': [round(ll_lo, 4), round(ll_hi, 4)],
        }
    return out


def filter_check(oos: pd.DataFrame, block: int) -> dict:
    """Directional hit rate of the ensemble's shown signals vs the calls the filter hid."""
    pred = oos['ensemble_pred'].to_numpy()
    directional = np.isin(pred, ['up', 'down'])
    hit = np.where(pred == 'up', oos['fwd_log_return'] > 0, oos['fwd_log_return'] < 0).astype(float)
    shown = (oos['ensemble_signal'] != 'none').to_numpy()
    hidden = directional & ~shown
    out = {'shown': int(shown.sum()), 'hidden': int(hidden.sum()),
           'shown_hit_rate': round(hit[shown].mean(), 4) if shown.any() else None,
           'hidden_hit_rate': round(hit[hidden].mean(), 4) if hidden.any() else None}
    if shown.sum() < 20 or hidden.sum() < 20:
        return {**out, 'diff_ci95': None, 'verdict': 'TOO FEW SIGNALS'}
    rng = np.random.default_rng(0)
    n = len(hit)
    up_hit = (oos['fwd_log_return'] > 0).to_numpy(float)
    diffs, skill = [], []
    for _ in range(1000):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        s_, h_ = shown[idx], hidden[idx]
        if s_.any() and h_.any():
            diffs.append(hit[idx][s_].mean() - hit[idx][h_].mean())
            skill.append(hit[idx][s_].mean() - up_hit[idx][s_].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    s_lo, s_hi = np.percentile(skill, [2.5, 97.5])
    out['diff_ci95'] = [round(float(lo), 4), round(float(hi), 4)]
    # A filter can look good by picking candles in rising stretches; compare with "always up" there too.
    out['vs_always_up_ci95'] = [round(float(s_lo), 4), round(float(s_hi), 4)]
    if lo > 0 and s_lo > 0:
        out['verdict'] = 'SHOWN BETTER'
    elif lo > 0:
        out['verdict'] = 'SHOWN BETTER THAN HIDDEN, BUT NOT THAN ALWAYS-UP'
    elif hi < 0:
        out['verdict'] = 'SHOWN WORSE'
    else:
        out['verdict'] = 'NO SIGNIFICANT DIFFERENCE'
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
    parser.add_argument('--experiment', choices=list(EXPERIMENTS), default='sr')
    parser.add_argument('--symbol', default='BTC')
    parser.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--folds', type=int, help='walk-forward folds (default: current setting)')
    parser.add_argument('--label-mode', choices=['ternary', 'binary'])
    parser.add_argument('--csv', help='use candles from a CSV instead of downloading')
    parser.add_argument('--csv-timeframe', default='1h', help='timeframe of the CSV candles')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON',
                        help='override a setting for both variants, e.g. --set \'neutral_thresholds={"1h": 0.0005}\'')
    parser.add_argument('--out', help='write the full comparison as JSON')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    settings_a, settings_b, prefix = EXPERIMENTS[args.experiment]

    cfg = get_config().app_config
    saved = asdict(cfg)
    cfg.output_dir = tempfile.mkdtemp(prefix=f'{args.experiment}_experiment_')
    if args.folds:
        cfg.walk_forward_splits = args.folds
    if args.label_mode:
        cfg.label_mode = args.label_mode
    if args.csv:
        fetcher = CsvFetcher(args.csv, args.csv_timeframe)
        cfg.use_btc_context = cfg.use_macro = False
    else:
        fetcher = CryptoDataFetcher()  # one instance: both variants see the same cached data
    for item in args.set:
        key, _, value = item.partition('=')
        cfg.update({key: json.loads(value)})
    common = {k: getattr(cfg, k) for k in asdict(cfg)}

    report = {'experiment': args.experiment, 'symbol': args.symbol, 'models': args.models,
              'source': args.csv or 'exchange', 'a': settings_a, 'b': settings_b, 'timeframes': {}}
    try:
        for tf in args.timeframes:
            runs = {}
            for variant, settings in (('a', settings_a), ('b', settings_b)):
                cfg.update({**common, **settings})
                service = PredictionService(fetcher=fetcher)
                print(f"[{tf}] running {variant}: {settings}...", file=sys.stderr)
                result = service.analyze(args.symbol, [tf], args.models)
                if tf not in result['timeframes']:
                    raise RuntimeError(f"{tf} failed: {result['errors'].get(tf)}")
                runs[variant] = (result['timeframes'][tf], service.oos[(result['symbol'], tf)])

            (res_a, oos_a), (res_b, oos_b) = runs['a'], runs['b']
            block = max(5, min(48, len(oos_a) // 50))
            cmp = paired_comparison(oos_a, oos_b, args.models, res_a['label']['classes'], block)
            cmp['verdict'] = verdict(cmp)
            cmp['backtest_return'] = {'a': res_a['backtest']['total_return'], 'b': res_b['backtest']['total_return']}
            cmp['directional_accuracy'] = {
                name: {'a': res_a['metrics'][name]['directional_accuracy'],
                       'b': res_b['metrics'][name]['directional_accuracy']}
                for name in args.models + ['ensemble']}
            cmp['shown_signals'] = {v: r['metrics']['ensemble']['signals'] for v, r in (('a', res_a), ('b', res_b))}
            cmp['filter_check'] = {'a': filter_check(oos_a, block), 'b': filter_check(oos_b, block)}
            cmp['notes'] = {'a': res_a['data']['notes'], 'b': res_b['data']['notes']}
            cmp['group_features_in_top10'] = [f['feature'] for f in res_b['feature_importance'][:10]
                                              if prefix and f['feature'].startswith(prefix)]
            report['timeframes'][tf] = cmp
            print_table(tf, cmp, args.models, args.experiment)
    finally:
        cfg.update({k: v for k, v in saved.items()})

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nSaved {args.out}", file=sys.stderr)
    return report


def print_table(tf: str, cmp: dict, models, experiment: str):
    print(f"\n=== {experiment} | {tf}: {cmp['paired_candles']} out-of-sample candles -> {cmp['verdict']} ===")
    print("   (A = without, B = with; accuracy over every test candle, paired)")
    print(f"{'model':<12}{'acc A':>8}{'acc B':>8}{'diff':>8}  {'95% CI':<18}{'p':>6}"
          f"{'dir A':>8}{'dir B':>8}{'logloss Δ':>11}")
    for name in models + ['ensemble']:
        m, d = cmp[name], cmp['directional_accuracy'][name]
        ci = f"[{m['accuracy_diff_ci95'][0]:+.3f}, {m['accuracy_diff_ci95'][1]:+.3f}]"
        fmt = lambda v: '--' if v is None else f"{v:.3f}"
        print(f"{name:<12}{m['accuracy_a']:>8.3f}{m['accuracy_b']:>8.3f}{m['accuracy_diff']:>+8.3f}  "
              f"{ci:<18}{m['accuracy_p_value']:>6.2f}{fmt(d['a']):>8}{fmt(d['b']):>8}{m['log_loss_diff']:>+11.4f}")
    bt = cmp['backtest_return']
    print(f"backtest (ensemble, shown signals only): A {bt['a']:+.1%}  vs  B {bt['b']:+.1%}")
    for v in ('a', 'b'):
        sig, fc = cmp['shown_signals'][v], cmp['filter_check'][v]
        pct = lambda x: '--' if x is None else f"{x:.1%}"
        print(f"shown signals {v.upper()}: {sig['n']} ({pct(sig['coverage'])} of candles), hit {pct(sig['directional_accuracy'])}"
              f" | always-up on same candles {pct(sig['always_up_same_candles'])}"
              f" | hidden calls hit {pct(fc['hidden_hit_rate'])} -> filter: {fc['verdict']}")
    if cmp['group_features_in_top10']:
        print(f"group features in GBM top-10: {', '.join(cmp['group_features_in_top10'])}")
    for v in ('a', 'b'):
        for note in cmp['notes'][v]:
            print(f"note {v.upper()}: {note}")


if __name__ == '__main__':
    main()
