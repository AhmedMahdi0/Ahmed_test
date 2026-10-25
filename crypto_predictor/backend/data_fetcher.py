"""
Market data fetching for CryptoPredictor.

Candles come from the Binance public klines API (native 1h / 4h / 1d candles,
long history, includes taker-buy volume). Yahoo Finance is used as a fallback
for candles and as the source of daily stock-index context.

All candle frames are indexed by the candle *open* time (UTC, tz-naive) and
only contain fully closed candles.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .config import get_config, get_timeframe, MARKET_INDICES, TIMEFRAMES

logger = logging.getLogger(__name__)

BINANCE_HOSTS = ['https://api.binance.com', 'https://data-api.binance.vision']
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base']
MACRO_TTL_SECONDS = 3600


class DataFetchError(Exception):
    """Raised when no usable data could be fetched."""


def _now_ms() -> int:
    return int(time.time() * 1000)


class CryptoDataFetcher:
    """Fetches candles and macro context, with an in-memory incremental cache."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.config = get_config()
        self.session = session or requests.Session()
        self._lock = threading.Lock()
        self._candles: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._candle_source: Dict[Tuple[str, str], str] = {}
        self._macro: Optional[pd.DataFrame] = None
        self._macro_fetched_at = 0.0

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str, timeframe: str, history_days: Optional[int] = None) -> pd.DataFrame:
        """
        Return closed candles for `symbol` on `timeframe`.

        Repeated calls only download candles newer than the cached ones.
        The frame has `attrs['source']` set to 'binance' or 'yfinance'.
        """
        spec = get_timeframe(timeframe)
        crypto = self.config.get_crypto(symbol)
        history_days = history_days or spec.history_days
        start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=history_days)
        key = (crypto.symbol, timeframe)

        with self._lock:
            cached = self._candles.get(key)
            source = self._candle_source.get(key)

        if cached is not None and cached.index[0] <= start + timedelta(minutes=spec.minutes) and source == 'binance':
            try:
                fresh = self._fetch_binance(crypto.binance_symbol, timeframe, cached.index[-1])
                data = pd.concat([cached, fresh])
                data = data[~data.index.duplicated(keep='last')].sort_index()
            except Exception as e:
                logger.warning("Incremental refresh failed for %s %s, using cached candles: %s",
                               crypto.symbol, timeframe, e)
                data = cached
        else:
            data, source = self._fetch_any(crypto, timeframe, start)

        data = data[data.index >= start]
        if data.empty:
            raise DataFetchError(f"No {timeframe} candles available for {symbol}")

        with self._lock:
            self._candles[key] = data
            self._candle_source[key] = source

        data = data.copy()
        data.attrs['source'] = source
        return data

    def _fetch_any(self, crypto, timeframe: str, start: datetime) -> Tuple[pd.DataFrame, str]:
        source = self.config.app_config.data_source
        errors = []
        if source in ('auto', 'binance'):
            try:
                return self._fetch_binance(crypto.binance_symbol, timeframe, start), 'binance'
            except Exception as e:
                errors.append(f"binance: {e}")
                logger.warning("Binance fetch failed for %s %s: %s", crypto.symbol, timeframe, e)
        if source in ('auto', 'yfinance'):
            try:
                return self._fetch_yfinance(crypto.yf_ticker, timeframe, start), 'yfinance'
            except Exception as e:
                errors.append(f"yfinance: {e}")
                logger.warning("Yahoo fetch failed for %s %s: %s", crypto.symbol, timeframe, e)
        raise DataFetchError(f"Could not fetch {timeframe} candles for {crypto.symbol} ({'; '.join(errors)})")

    def _binance_get(self, path: str, params: Dict) -> list:
        last_error = None
        for host in BINANCE_HOSTS:
            try:
                resp = self.session.get(f"{host}{path}", params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(2)
                    resp = self.session.get(f"{host}{path}", params=params, timeout=20)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_error = e
        raise DataFetchError(f"Binance request failed: {last_error}")

    def _fetch_binance(self, binance_symbol: str, timeframe: str, start: datetime) -> pd.DataFrame:
        spec = get_timeframe(timeframe)
        start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        rows = []
        while True:
            batch = self._binance_get('/api/v3/klines', {
                'symbol': binance_symbol, 'interval': timeframe, 'startTime': start_ms, 'limit': 1000,
            })
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            start_ms = batch[-1][0] + spec.millis
            time.sleep(0.1)

        now = _now_ms()
        rows = [r for r in rows if r[6] < now]  # drop the still-open candle
        if not rows:
            raise DataFetchError(f"Binance returned no closed candles for {binance_symbol}")

        df = pd.DataFrame(rows, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore',
        ])
        df.index = pd.to_datetime(df['open_time'], unit='ms')
        df.index.name = 'time'
        return df[OHLCV_COLUMNS].astype(float)

    def _fetch_yfinance(self, ticker: str, timeframe: str, start: datetime) -> pd.DataFrame:
        import yfinance as yf

        spec = get_timeframe(timeframe)
        interval = '1d' if timeframe == '1d' else '1h'
        if interval == '1h':
            # Yahoo only serves ~730 days of hourly candles.
            start = max(start, datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=729))
        raw = yf.download(ticker, start=start.strftime('%Y-%m-%d'), interval=interval,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise DataFetchError(f"Yahoo returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']].astype(float)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        df.index.name = 'time'

        if timeframe == '4h':
            grouped = df.resample('4h', origin='epoch', label='left', closed='left')
            df = grouped.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            df['n'] = grouped['close'].count()
            df = df[df['n'] > 0].drop(columns='n')

        # Keep only candles that have closed.
        now = pd.Timestamp.now(tz='UTC').tz_localize(None)
        df = df[df.index + pd.Timedelta(minutes=spec.minutes) <= now].dropna(subset=['close'])
        df['quote_volume'] = df['volume'] * df['close']
        df['trades'] = np.nan
        df['taker_buy_base'] = np.nan
        return df[OHLCV_COLUMNS]

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Last traded price (Binance ticker), or None when unavailable."""
        try:
            crypto = self.config.get_crypto(symbol)
            data = self._binance_get('/api/v3/ticker/price', {'symbol': crypto.binance_symbol})
            return float(data['price'])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Daily macro context
    # ------------------------------------------------------------------

    def fetch_macro(self) -> pd.DataFrame:
        """
        Daily macro frame indexed by calendar date: index closes (SP500, NASDAQ,
        DXY, VIX) and the crypto Fear & Greed index (FGI). Missing sources are
        simply absent; an empty frame is returned if nothing is reachable.
        """
        with self._lock:
            if self._macro is not None and time.time() - self._macro_fetched_at < MACRO_TTL_SECONDS:
                return self._macro

        parts = []
        fgi = self._fetch_fear_greed()
        if not fgi.empty:
            parts.append(fgi)
        indices = self._fetch_indices()
        if not indices.empty:
            parts.append(indices)
        macro = pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()

        with self._lock:
            self._macro = macro
            self._macro_fetched_at = time.time()
        return macro

    def _fetch_fear_greed(self) -> pd.DataFrame:
        try:
            resp = self.session.get("https://api.alternative.me/fng/?limit=0", timeout=30)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json()['data'])
            df.index = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s').normalize()
            series = pd.to_numeric(df['value'], errors='coerce').rename('FGI')
            return series[~series.index.duplicated()].sort_index().to_frame()
        except Exception as e:
            logger.warning("Fear & Greed index unavailable: %s", e)
            return pd.DataFrame()

    def _fetch_indices(self) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            logger.info("yfinance not installed; skipping stock-index features")
            return pd.DataFrame()

        max_days = max(spec.history_days for spec in TIMEFRAMES.values())
        start = (datetime.now(timezone.utc) - timedelta(days=max_days + 30)).strftime('%Y-%m-%d')
        out = {}
        for ticker, name in MARKET_INDICES.items():
            try:
                raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
                if raw is None or raw.empty:
                    continue
                close = raw['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                if close.index.tz is not None:
                    close.index = close.index.tz_localize(None)
                out[name] = close.astype(float)
            except Exception as e:
                logger.warning("Could not download %s: %s", name, e)
        return pd.DataFrame(out)
