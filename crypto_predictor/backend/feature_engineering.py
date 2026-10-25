"""
Feature engineering and labelling for direction prediction.

Every feature on row t is computed only from information available when
candle t closes. The label on row t describes the move from close[t] to
close[t + horizon], so it is never visible to the features.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import get_config, get_timeframe

logger = logging.getLogger(__name__)

TERNARY_CLASSES = ['down', 'neutral', 'up']
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

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.y.notna().to_numpy()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


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
                     macro: Optional[pd.DataFrame] = None) -> Dataset:
        cfg = self.config.app_config
        horizon = cfg.horizon_bars
        threshold = cfg.threshold_for(timeframe)
        classes = class_names(cfg.label_mode)

        parts = [self.candle_features(ohlcv), self.time_features(ohlcv.index, timeframe)]
        if btc is not None and not btc.empty:
            parts.append(self.btc_features(btc, ohlcv.index))
        if macro is not None and not macro.empty:
            parts.append(self.macro_features(macro, ohlcv.index, timeframe))
        X = pd.concat(parts, axis=1)

        # Drop columns that are mostly missing (e.g. a macro source with a short history).
        coverage = X.notna().mean()
        X = X.loc[:, coverage >= 0.8]
        valid = X.notna().all(axis=1)
        X = X[valid]

        close = ohlcv['close'].reindex(X.index)
        full_close = ohlcv['close']
        fwd = np.log(full_close.shift(-horizon) / full_close).reindex(X.index)
        past = np.log(full_close / full_close.shift(horizon)).reindex(X.index)
        next_ret = (full_close.shift(-1) / full_close - 1).reindex(X.index)

        y = classify_returns(fwd, threshold, cfg.label_mode)
        persistence = classify_returns(past, threshold, cfg.label_mode)

        logger.info("Dataset %s: %d rows, %d features, %d labelled", timeframe, len(X), X.shape[1], int(y.notna().sum()))
        return Dataset(X=X, y=y, fwd_return=fwd, next_return=next_ret, persistence=persistence,
                       close=close, classes=classes, timeframe=timeframe, threshold=threshold, horizon=horizon)
