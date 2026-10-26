"""
Feature engineering and labelling for direction prediction.

Every feature on row t is computed only from information available when
candle t closes. The label on row t describes the move from close[t] to
close[t + horizon], so it is never visible to the features.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import get_config, get_timeframe

logger = logging.getLogger(__name__)

TERNARY_CLASSES = ['down', 'neutral', 'up']
# Higher timeframes used as context for each model timeframe ('1w' is built from 1d candles).
HTF_MAP = {'1h': ['4h', '1d'], '4h': ['1d', '1w'], '1d': ['1w']}
HTF_MINUTES = {'4h': 240, '1d': 1440, '1w': 10080}
HTF_COLUMNS = ['ret_1', 'ret_3', 'rsi', 'macd_hist', 'dist_ema20', 'dist_ema50', 'bb_pctb', 'atr_pct', 'vol_12']
BINARY_CLASSES = ['down', 'up']


def class_names(label_mode: str) -> List[str]:
    return TERNARY_CLASSES if label_mode == 'ternary' else BINARY_CLASSES


@dataclass
class Dataset:
    """Model-ready data for one symbol/timeframe."""
    X: pd.DataFrame              # features, one row per closed candle (no NaNs)
    y: pd.Series                 # class index; NaN where the future is not known yet
    fwd_return: pd.Series        # log return over the label horizon (NaN at the tail)
    next_return: pd.Series       # simple return of the next candle (for backtests)
    persistence: pd.Series       # class of the *past* move, a naive baseline prediction
    close: pd.Series
    classes: List[str]
    timeframe: str
    threshold: float
    horizon: int
    notes: List[str] = field(default_factory=list)   # dropped columns, trimmed history, ...

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.y.notna().to_numpy()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def period_start(index: pd.DatetimeIndex, htf: str) -> pd.DatetimeIndex:
    """Open time of the higher-timeframe candle each timestamp belongs to (Binance boundaries, UTC)."""
    if htf == '1w':
        days = index.normalize()
        return days - pd.to_timedelta(days.day_of_week, unit='D')   # weeks start Monday 00:00
    return index.floor(f'{HTF_MINUTES[htf]}min')


def resample_ohlcv(df: pd.DataFrame, htf: str) -> pd.DataFrame:
    """Aggregate candles into a higher timeframe (indexed by the higher candle's open time)."""
    g = df.groupby(period_start(df.index, htf))
    out = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                        'close': g['close'].last(), 'volume': g['volume'].sum()})
    out['quote_volume'] = g['quote_volume'].sum() if 'quote_volume' in df else np.nan
    out['trades'] = np.nan
    out['taker_buy_base'] = g['taker_buy_base'].sum(min_count=1) if 'taker_buy_base' in df else np.nan
    out.index.name = 'time'
    return out


def _asof(index_times: pd.DatetimeIndex, frame: pd.DataFrame, tolerance=None) -> pd.DataFrame:
    """For each time, the last row of `frame` whose index is <= that time."""
    left = pd.DataFrame({'t': pd.DatetimeIndex(index_times).astype('datetime64[ns]')})
    right = frame.copy()
    right.index = pd.DatetimeIndex(right.index).astype('datetime64[ns]')
    right = right.rename_axis('t_right').reset_index().sort_values('t_right')
    merged = pd.merge_asof(left, right, left_on='t', right_on='t_right', direction='backward',
                           tolerance=tolerance)
    return merged.drop(columns=['t', 't_right'])


def classify_returns(returns: pd.Series, threshold: float, label_mode: str) -> pd.Series:
    """Map log returns to class indices (NaN stays NaN)."""
    if label_mode == 'ternary':
        labels = np.where(returns > threshold, 2, np.where(returns < -threshold, 0, 1)).astype(float)
    else:
        labels = np.where(returns > 0, 1, 0).astype(float)
    return pd.Series(labels, index=returns.index).where(returns.notna())


class FeatureEngineer:
    """Builds stationary, leakage-free features and direction labels."""

    def __init__(self):
        self.config = get_config()

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def candle_features(self, df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
        cfg = self.config.app_config
        close, high, low, open_ = df['close'], df['high'], df['low'], df['open']
        logret = np.log(close).diff()
        f = pd.DataFrame(index=df.index)

        f['ret_1'] = logret
        for k in (2, 3, 6, 12, 24):
            f[f'ret_{k}'] = np.log(close / close.shift(k))
        for lag in range(1, cfg.return_lags + 1):
            f[f'ret_lag{lag}'] = logret.shift(lag)
        for w in (12, 24, 48):
            f[f'vol_{w}'] = logret.rolling(w).std()
        f['vol_ratio'] = f['vol_12'] / f['vol_48']

        f['rsi'] = _rsi(close, cfg.rsi_window) / 100.0
        ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
        macd = (ema_fast - ema_slow) / close
        macd_signal = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
        f['macd'] = macd
        f['macd_hist'] = macd - macd_signal

        prev_close = close.shift(1)
        true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        f['atr_pct'] = true_range.rolling(cfg.atr_window).mean() / close

        bb_mid = close.rolling(cfg.bb_window).mean()
        bb_std = close.rolling(cfg.bb_window).std()
        f['bb_pctb'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)
        f['bb_width'] = 4 * bb_std / bb_mid

        for span in (20, 50, 200):
            f[f'dist_ema{span}'] = close / close.ewm(span=span, adjust=False, min_periods=span).mean() - 1

        rng = (high - low).replace(0, np.nan)
        f['body'] = (close - open_) / open_
        f['range_pct'] = (high - low) / open_
        f['upper_wick'] = (high - np.maximum(open_, close)) / rng
        f['lower_wick'] = (np.minimum(open_, close) - low) / rng
        f['close_pos'] = (close - low) / rng

        log_vol = np.log1p(df['volume'])
        f['volume_z'] = (log_vol - log_vol.rolling(48).mean()) / log_vol.rolling(48).std()
        f['volume_chg'] = log_vol.diff()
        if df['taker_buy_base'].notna().any():
            f['taker_buy_ratio'] = (df['taker_buy_base'] / df['volume'].replace(0, np.nan)).fillna(0.5)
        else:
            f['taker_buy_ratio'] = 0.5

        f = f.replace([np.inf, -np.inf], np.nan)
        if prefix:
            f = f.add_prefix(prefix)
        return f

    def support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Distance to and strength of nearby support / resistance levels.

        Levels come from swing points: candle i is a swing high if its high is the
        highest of the k candles on each side. That is only knowable k candles
        later, so a swing point becomes a level at i + k, never earlier. Highs and
        lows both count as levels (old resistance can become support).
        Distances are in ATRs so they mean the same thing for any price or timeframe.
        """
        cfg = self.config.app_config
        k, lookback = cfg.sr_pivot_bars, cfg.sr_lookback
        high, low, close = df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy()
        n = len(df)

        prev_close = df['close'].shift(1)
        true_range = pd.concat([df['high'] - df['low'], (df['high'] - prev_close).abs(),
                                (df['low'] - prev_close).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(cfg.atr_window).mean().to_numpy()

        win = 2 * k + 1
        is_high = (df['high'] == df['high'].rolling(win, center=True).max()).to_numpy()
        is_low = (df['low'] == df['low'].rolling(win, center=True).min()).to_numpy()
        hi_idx, lo_idx = np.flatnonzero(is_high), np.flatnonzero(is_low)
        pivot_idx = np.concatenate([hi_idx, lo_idx])
        pivot_price = np.concatenate([high[hi_idx], low[lo_idx]])
        order = np.argsort(pivot_idx, kind='stable')
        pivot_idx, pivot_price = pivot_idx[order], pivot_price[order]
        confirmed_at = pivot_idx + k

        cap = 10.0  # "no level within reach", in ATRs
        res_dist = np.full(n, np.nan)
        sup_dist = np.full(n, np.nan)
        res_strength = np.full(n, np.nan)
        sup_strength = np.full(n, np.nan)
        res_level = np.full(n, np.nan)
        sup_level = np.full(n, np.nan)
        start = 0
        stop = 0
        for t in range(n):
            if np.isnan(atr[t]) or atr[t] <= 0:
                continue
            while stop < len(confirmed_at) and confirmed_at[stop] <= t:
                stop += 1
            while start < stop and pivot_idx[start] < t - lookback:
                start += 1
            levels = pivot_price[start:stop]
            zone = 0.5 * atr[t]
            above = levels[levels > close[t]]
            below = levels[levels < close[t]]
            if above.size:
                r = above.min()
                res_level[t] = r
                res_dist[t] = min((r - close[t]) / atr[t], cap)
                res_strength[t] = np.count_nonzero(np.abs(levels - r) <= zone)
            else:
                res_dist[t], res_strength[t] = cap, 0
            if below.size:
                s = below.max()
                sup_level[t] = s
                sup_dist[t] = min((close[t] - s) / atr[t], cap)
                sup_strength[t] = np.count_nonzero(np.abs(levels - s) <= zone)
            else:
                sup_dist[t], sup_strength[t] = cap, 0

        f = pd.DataFrame(index=df.index)
        f['sr_res_dist_atr'] = res_dist
        f['sr_sup_dist_atr'] = sup_dist
        f['sr_res_strength'] = res_strength
        f['sr_sup_strength'] = sup_strength
        span = res_dist + sup_dist
        f['sr_position'] = np.where(span > 0, sup_dist / span, 0.5)
        f.loc[np.isnan(span), 'sr_position'] = np.nan

        # Events relative to the levels that existed one candle earlier.
        prev_res, prev_sup = pd.Series(res_level).shift(1).to_numpy(), pd.Series(sup_level).shift(1).to_numpy()
        prev_atr = pd.Series(atr).shift(1).to_numpy()
        with np.errstate(invalid='ignore'):
            f['sr_break_up'] = (close > prev_res).astype(float)
            f['sr_break_down'] = (close < prev_sup).astype(float)
            f['sr_reject_res'] = ((high >= prev_res - 0.2 * prev_atr) & (close < prev_res)).astype(float)
            f['sr_bounce_sup'] = ((low <= prev_sup + 0.2 * prev_atr) & (close > prev_sup)).astype(float)

        # Simple range extremes (Donchian channels), also in ATRs.
        atr_s = pd.Series(atr, index=df.index)
        for w in (20, 50, 100):
            f[f'sr_dist_high_{w}'] = (df['high'].rolling(w).max() - df['close']) / atr_s
            f[f'sr_dist_low_{w}'] = (df['close'] - df['low'].rolling(w).min()) / atr_s
        return f.replace([np.inf, -np.inf], np.nan)

    def htf_features(self, ohlcv: pd.DataFrame, timeframe: str, higher: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Context from higher timeframes:
          * the last *closed* higher candle's indicators (it becomes visible when it closes),
          * the higher candle that is still forming, rebuilt from this timeframe's candles
            up to now: return since it opened and where price sits in its range so far.
        """
        minutes = get_timeframe(timeframe).minutes
        close_time = ohlcv.index + pd.Timedelta(minutes=minutes)
        f = pd.DataFrame(index=ohlcv.index)
        for name, candles in higher.items():
            if candles is None or candles.empty:
                continue
            feats = self.candle_features(candles)[HTF_COLUMNS]
            feats.index = candles.index + pd.Timedelta(minutes=HTF_MINUTES[name])   # known at its close
            aligned = _asof(close_time, feats)
            aligned.index = ohlcv.index
            f = f.join(aligned.add_prefix(f'htf_{name}_'))

            group = period_start(ohlcv.index, name)
            p_open = ohlcv['open'].groupby(group).transform('first')
            p_high = ohlcv['high'].groupby(group).cummax()
            p_low = ohlcv['low'].groupby(group).cummin()
            rng = (p_high - p_low).replace(0, np.nan)
            f[f'htf_{name}_partial_ret'] = np.log(ohlcv['close'] / p_open)
            f[f'htf_{name}_partial_pos'] = ((ohlcv['close'] - p_low) / rng).fillna(0.5)
        return f.replace([np.inf, -np.inf], np.nan)

    def futures_features(self, futures: Dict, ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Funding, perpetual premium and open-interest / positioning features."""
        minutes = get_timeframe(timeframe).minutes
        close_time = ohlcv.index + pd.Timedelta(minutes=minutes)
        f = pd.DataFrame(index=ohlcv.index)

        funding = futures.get('funding')
        if funding is not None and not funding.empty:
            fr = funding['funding_rate'] * 1e4                     # basis points per settlement
            ff = pd.DataFrame({
                'fut_funding': fr,
                'fut_funding_mean_3': fr.rolling(3).mean(),         # ~1 day
                'fut_funding_mean_21': fr.rolling(21).mean(),       # ~1 week
                'fut_funding_z': (fr - fr.rolling(90).mean()) / fr.rolling(90).std(),
            })
            aligned = _asof(close_time, ff, tolerance=pd.Timedelta(hours=9))   # settles every 8h
            aligned.index = ohlcv.index
            f = f.join(aligned)

        premium = futures.get('premium')
        if premium is not None and not premium.empty:
            prem = premium['premium'].reindex(ohlcv.index) * 1e4    # same candles, known at close
            f['fut_premium'] = prem
            f['fut_premium_chg'] = prem.diff()
            f['fut_premium_mean_24'] = prem.rolling(24).mean()

        metrics = futures.get('metrics')
        if metrics is not None and not metrics.empty:
            m = metrics.copy()
            # Taker buy/sell is a flow: average it over the candle rather than taking one 5-minute snapshot.
            m['taker_ls'] = m['taker_ls'].rolling(f'{minutes}min').mean()
            tolerance = pd.Timedelta(minutes=max(2 * minutes, 180))
            aligned = _asof(close_time, m, tolerance=tolerance)
            aligned.index = ohlcv.index
            oi = np.log(aligned['open_interest'])
            f['fut_oi_chg_1'] = oi.diff()
            f['fut_oi_chg_24'] = oi.diff(24)
            f['fut_oi_z'] = (oi - oi.rolling(168).mean()) / oi.rolling(168).std()
            for col in ('top_trader_ls', 'global_ls', 'taker_ls'):
                level = np.log(aligned[col])
                f[f'fut_{col}'] = level
                f[f'fut_{col}_chg_24'] = level.diff(24)
        return f.replace([np.inf, -np.inf], np.nan)

    def time_features(self, index: pd.DatetimeIndex, timeframe: str) -> pd.DataFrame:
        f = pd.DataFrame(index=index)
        dow = index.day_of_week
        f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        f['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        if timeframe != '1d':
            hour = index.hour
            f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        return f

    def btc_features(self, btc: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        logret = np.log(btc['close']).diff()
        f = pd.DataFrame(index=btc.index)
        f['btc_ret_1'] = logret
        f['btc_ret_6'] = logret.rolling(6).sum()
        f['btc_ret_24'] = logret.rolling(24).sum()
        f['btc_vol_24'] = logret.rolling(24).std()
        f['btc_rsi'] = _rsi(btc['close'], self.config.app_config.rsi_window) / 100.0
        # Same open times on both series, so a plain reindex keeps alignment exact.
        return f.reindex(index)

    def macro_features(self, macro: pd.DataFrame, index: pd.DatetimeIndex, timeframe: str) -> pd.DataFrame:
        """
        Align daily macro data so a candle only sees values from days that had
        fully finished before the candle closed.
        """
        if macro is None or macro.empty:
            return pd.DataFrame(index=index)

        daily = macro.copy()
        daily.index = pd.DatetimeIndex(daily.index).normalize()
        daily = daily[~daily.index.duplicated(keep='last')].sort_index()

        f = pd.DataFrame(index=daily.index)
        for col in ('SP500', 'NASDAQ', 'DXY'):
            if col in daily:
                s = daily[col].dropna()
                f[f'{col.lower()}_ret'] = np.log(s).diff().reindex(daily.index)
        if 'VIX' in daily:
            vix = daily['VIX'].dropna()
            f['vix_log'] = np.log(vix).reindex(daily.index)
            f['vix_chg'] = np.log(vix).diff().reindex(daily.index)
        if 'FGI' in daily:
            fgi = daily['FGI'].dropna()
            f['fgi'] = (fgi / 100.0).reindex(daily.index)
            f['fgi_chg_1'] = (fgi.diff() / 100.0).reindex(daily.index)
            f['fgi_chg_7'] = (fgi.diff(7) / 100.0).reindex(daily.index)

        # Calendar-daily with forward fill (weekends/holidays keep the last value).
        f = f.reindex(pd.date_range(f.index.min(), f.index.max(), freq='D')).ffill()
        # A value for day D is only known once day D is over.
        f.index = (f.index + pd.Timedelta(days=1)).astype('datetime64[ns]')

        close_time = (index + pd.Timedelta(minutes=get_timeframe(timeframe).minutes)).astype('datetime64[ns]')
        aligned = pd.merge_asof(
            pd.DataFrame({'close_time': close_time}),
            f.rename_axis('available_at').reset_index(),
            left_on='close_time', right_on='available_at', direction='backward',
        )
        aligned.index = index
        return aligned.drop(columns=['close_time', 'available_at'])

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def make_dataset(self,
                     ohlcv: pd.DataFrame,
                     timeframe: str,
                     btc: Optional[pd.DataFrame] = None,
                     macro: Optional[pd.DataFrame] = None,
                     higher: Optional[Dict[str, pd.DataFrame]] = None,
                     futures: Optional[Dict] = None) -> Dataset:
        cfg = self.config.app_config
        horizon = cfg.horizon_bars
        threshold = cfg.threshold_for(timeframe)
        classes = class_names(cfg.label_mode)

        parts = [self.candle_features(ohlcv), self.time_features(ohlcv.index, timeframe)]
        if cfg.use_support_resistance:
            parts.append(self.support_resistance_features(ohlcv))
        if btc is not None and not btc.empty:
            parts.append(self.btc_features(btc, ohlcv.index))
        if macro is not None and not macro.empty:
            parts.append(self.macro_features(macro, ohlcv.index, timeframe))
        X = pd.concat(parts, axis=1)
        notes = []

        # Drop columns that are mostly missing (e.g. a macro source with a short history).
        coverage = X.notna().mean()
        X = X.loc[:, coverage >= 0.8]

        # Opt-in groups: keep their columns and start the dataset once they have data, rather than
        # silently dropping them. Only columns with less than half the history are given up.
        opt_in = []
        if cfg.use_htf_context and higher:
            opt_in.append(('higher-timeframe', self.htf_features(ohlcv, timeframe, higher)))
        if cfg.use_futures and futures:
            opt_in.append(('futures', self.futures_features(futures, ohlcv, timeframe)))
        for group, frame in opt_in:
            cov = frame.notna().mean()
            short = cov[cov < 0.5].index.tolist()
            if short:
                notes.append(f"{group}: dropped {len(short)} columns with under 50% history ({', '.join(short[:4])}"
                             f"{'...' if len(short) > 4 else ''})")
            X = X.join(frame.drop(columns=short))

        valid = X.notna().all(axis=1)
        X = X[valid]
        if len(X) and opt_in and X.index[0] > ohlcv.index[0] + pd.Timedelta(days=30):
            notes.append(f"history starts {X.index[0]:%Y-%m-%d} (when all selected data sources are available)")
        if len(X) and X.index[-1] < ohlcv.index[-1]:
            notes.append(f"latest candle {ohlcv.index[-1]:%Y-%m-%d %H:%M} lacks some features; "
                         f"signal is based on {X.index[-1]:%Y-%m-%d %H:%M}")

        close = ohlcv['close'].reindex(X.index)
        full_close = ohlcv['close']
        fwd = np.log(full_close.shift(-horizon) / full_close).reindex(X.index)
        past = np.log(full_close / full_close.shift(horizon)).reindex(X.index)
        next_ret = (full_close.shift(-1) / full_close - 1).reindex(X.index)

        y = classify_returns(fwd, threshold, cfg.label_mode)
        persistence = classify_returns(past, threshold, cfg.label_mode)

        logger.info("Dataset %s: %d rows, %d features, %d labelled", timeframe, len(X), X.shape[1], int(y.notna().sum()))
        return Dataset(X=X, y=y, fwd_return=fwd, next_return=next_ret, persistence=persistence,
                       close=close, classes=classes, timeframe=timeframe, threshold=threshold, horizon=horizon,
                       notes=notes)
