import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config as config_module  # noqa: E402
from backend.config import get_timeframe  # noqa: E402


def make_candles(n: int, timeframe: str, seed: int = 0, phi: float = 0.2, start='2022-01-01') -> pd.DataFrame:
    """Synthetic candles whose returns are AR(1), so direction is partly predictable."""
    rng = np.random.default_rng(seed)
    spec = get_timeframe(timeframe)
    eps = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    volume = rng.lognormal(10, 0.3, n)
    idx = pd.date_range(start, periods=n, freq=f'{spec.minutes}min')
    return pd.DataFrame({
        'open': open_, 'high': np.maximum(open_, close) + spread, 'low': np.minimum(open_, close) - spread,
        'close': close, 'volume': volume, 'quote_volume': volume * close,
        'trades': rng.integers(100, 1000, n).astype(float),
        'taker_buy_base': volume * rng.uniform(0.4, 0.6, n),
    }, index=idx.rename('time'))


class FakeFetcher:
    """Stands in for CryptoDataFetcher (no network)."""

    def __init__(self, n=1500):
        self.n = n
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, history_days=None):
        self.calls.append((symbol, timeframe))
        df = make_candles(self.n, timeframe, seed=zlib.crc32(f'{symbol}-{timeframe}'.encode()) % 1000)
        df.attrs['source'] = 'fake'
        return df

    def fetch_macro(self):
        days = pd.date_range('2021-06-01', '2027-01-01', freq='D')
        rng = np.random.default_rng(1)
        return pd.DataFrame({'FGI': rng.integers(10, 90, len(days)).astype(float)}, index=days)

    def get_latest_price(self, symbol):
        return 123.0


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point config and model output at a temp dir and use fast settings."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    config_module.reset_config()
    cfg = config_module.get_config().app_config
    cfg.output_dir = str(tmp_path / 'out')
    cfg.epochs = 3
    cfg.gbm_estimators = 60
    cfg.walk_forward_splits = 3
    yield config_module.get_config()
    config_module.reset_config()
