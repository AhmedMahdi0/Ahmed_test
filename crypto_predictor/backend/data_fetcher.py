"""
Market data fetching for CryptoPredictor.

Candles come from the Binance public klines API (native 1h / 4h / 1d candles,
long history, includes taker-buy volume). Yahoo Finance is used as a fallback
for candles and as the source of daily stock-index context. Perpetual-futures
context (funding, premium index, open interest, long/short ratios) comes from
the Binance USD-M futures API and the data.binance.vision daily archives.

All candle frames are indexed by the candle *open* time (UTC, tz-naive) and
only contain fully closed candles.
"""

import io
import logging
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .config import get_config, get_timeframe, MARKET_INDICES, TIMEFRAMES

logger = logging.getLogger(__name__)

BINANCE_HOSTS = ['https://api.binance.com', 'https://data-api.binance.vision']
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base']
MACRO_TTL_SECONDS = 3600

FAPI_HOST = 'https://fapi.binance.com'
VISION_METRICS_URL = 'https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day}.zip'
METRICS_FIRST_DAY = date(2021, 12, 1)   # Binance's open-interest archive does not go further back
FUTURES_TTL_SECONDS = 600
METRIC_COLUMNS = ['open_interest', 'top_trader_ls', 'global_ls', 'taker_ls']


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
        self._futures: Dict[Tuple[str, str], Tuple[float, Dict]] = {}

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

    # ------------------------------------------------------------------
    # Perpetual futures context
    # ------------------------------------------------------------------

    def fetch_futures(self, symbol: str, timeframe: str, history_days: Optional[int] = None) -> Dict:
        """
        Futures context for `symbol`, as a dict that may hold:
          'funding'  - funding rate per settlement, indexed by settlement time
          'premium'  - premium-index candle closes on `timeframe`, indexed by open time
          'metrics'  - open interest and long/short ratios, indexed by the time each value
                       became available (already shifted to be conservative)
        plus 'warnings' for the parts that could not be fetched. Raises DataFetchError
        only when nothing at all is available (e.g. the coin has no USDT perpetual).
        """
        crypto = self.config.get_crypto(symbol)
        spec = get_timeframe(timeframe)
        key = (crypto.symbol, timeframe)
        with self._lock:
            cached = self._futures.get(key)
        if cached and time.time() - cached[0] < FUTURES_TTL_SECONDS:
            return cached[1]

        start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=history_days or spec.history_days)
        sym = crypto.binance_symbol
        out, warnings = {}, []
        for name, fn in (('funding', lambda: self._fetch_funding(sym, start)),
                         ('premium', lambda: self._fetch_premium(sym, timeframe, start)),
                         ('metrics', lambda: self._fetch_metrics(sym, start))):
            try:
                frame = fn()
                if frame is not None and not frame.empty:
                    out[name] = frame
                else:
                    warnings.append(f"{name}: no data")
            except Exception as e:
                warnings.append(f"{name}: {e}")
                logger.warning("Futures %s unavailable for %s: %s", name, symbol, e)
        if not out:
            raise DataFetchError(f"No futures data for {symbol} ({'; '.join(warnings)})")
        out['warnings'] = warnings
        with self._lock:
            self._futures[key] = (time.time(), out)
        return out

    def _fapi_get(self, path: str, params: Dict) -> list:
        resp = self.session.get(f"{FAPI_HOST}{path}", params=params, timeout=20)
        if resp.status_code == 429:
            time.sleep(2)
            resp = self.session.get(f"{FAPI_HOST}{path}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _fetch_funding(self, sym: str, start: datetime) -> pd.DataFrame:
        start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        rows = []
        while True:
            batch = self._fapi_get('/fapi/v1/fundingRate', {'symbol': sym, 'startTime': start_ms, 'limit': 1000})
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            start_ms = int(batch[-1]['fundingTime']) + 1
            time.sleep(0.1)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(pd.to_numeric(df['fundingTime']), unit='ms')
        df = df[~df.index.duplicated(keep='last')].sort_index()
        return pd.DataFrame({'funding_rate': pd.to_numeric(df['fundingRate'], errors='coerce')}, index=df.index)

    def _fetch_premium(self, sym: str, timeframe: str, start: datetime) -> pd.DataFrame:
        spec = get_timeframe(timeframe)
        start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        rows = []
        while True:
            batch = self._fapi_get('/fapi/v1/premiumIndexKlines',
                                   {'symbol': sym, 'interval': timeframe, 'startTime': start_ms, 'limit': 1500})
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1500:
                break
            start_ms = batch[-1][0] + spec.millis
            time.sleep(0.1)
        now = _now_ms()
        rows = [r for r in rows if r[6] < now]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame({'premium': [float(r[4]) for r in rows]},
                          index=pd.to_datetime([r[0] for r in rows], unit='ms'))
        return df[~df.index.duplicated(keep='last')].sort_index()

    def _metrics_cache_dir(self, sym: str) -> Path:
        path = Path(self.config.app_config.output_dir) / 'cache' / 'futures_metrics' / sym
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _fetch_metrics_day(self, sym: str, day: date, cache_dir: Path) -> Optional[pd.DataFrame]:
        csv_path = cache_dir / f"{day.isoformat()}.csv"
        missing_path = cache_dir / f"{day.isoformat()}.missing"
        if csv_path.exists():
            return pd.read_csv(csv_path, parse_dates=['available_at'])
        if missing_path.exists():
            return None
        url = VISION_METRICS_URL.format(sym=sym, day=day.isoformat())
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            # Archives appear a day or two late; only remember a miss once it is clearly final.
            if day < datetime.now(timezone.utc).date() - timedelta(days=3):
                missing_path.touch()
            return None
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = pd.read_csv(zf.open(zf.namelist()[0]))
        df = parse_vision_metrics(raw)
        df.to_csv(csv_path, index=False)
        return df

    def _fetch_metrics(self, sym: str, start: datetime) -> pd.DataFrame:
        first = max(start.date(), METRICS_FIRST_DAY)
        last = datetime.now(timezone.utc).date() - timedelta(days=1)
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        cache_dir = self._metrics_cache_dir(sym)
        frames: List[pd.DataFrame] = []
        if days:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for df in pool.map(lambda d: self._fetch_metrics_day(sym, d, cache_dir), days):
                    if df is not None and not df.empty:
                        frames.append(df)
        try:
            recent = self._fetch_recent_metrics(sym)
            if not recent.empty:
                frames.append(recent)
        except Exception as e:
            logger.warning("Recent open-interest data unavailable for %s: %s", sym, e)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df['available_at'] = pd.to_datetime(df['available_at'])
        # Archive rows come first, so they win over the API's coarser hourly values.
        df = df.drop_duplicates(subset='available_at', keep='first').set_index('available_at').sort_index()
        return df[METRIC_COLUMNS].astype(float)

    def _fetch_recent_metrics(self, sym: str) -> pd.DataFrame:
        """Last ~20 days at 1h resolution from the API (the archive lags by a day or two)."""
        params = {'symbol': sym, 'period': '1h', 'limit': 500}
        sources = {
            'open_interest': ('/futures/data/openInterestHist', 'sumOpenInterest'),
            'top_trader_ls': ('/futures/data/topLongShortPositionRatio', 'longShortRatio'),
            'global_ls': ('/futures/data/globalLongShortAccountRatio', 'longShortRatio'),
            'taker_ls': ('/futures/data/takerlongshortRatio', 'buySellRatio'),
        }
        cols = {}
        for col, (path, field) in sources.items():
            rows = self._fapi_get(path, params)
            if rows:
                cols[col] = pd.Series([float(r[field]) for r in rows],
                                      index=pd.to_datetime([int(r['timestamp']) for r in rows], unit='ms'))
        if not cols:
            return pd.DataFrame()
        df = pd.DataFrame(cols)
        # The API does not document whether a timestamp opens or closes its hour; assume the
        # later of the two so a value is never used before it can have been known.
        df.index = df.index + pd.Timedelta(hours=1)
        return df.rename_axis('available_at').reset_index()


def parse_vision_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a data.binance.vision `metrics` CSV (5-minute snapshots)."""
    names = ['create_time', 'symbol', 'sum_open_interest', 'sum_open_interest_value',
             'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
             'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
    if 'create_time' not in raw.columns:   # some archives have no header row
        header = pd.DataFrame([raw.columns.tolist()], columns=names[:len(raw.columns)])
        raw = pd.concat([header, raw.set_axis(names[:len(raw.columns)], axis=1)], ignore_index=True)
    out = pd.DataFrame({
        # A snapshot stamped T summarises the 5 minutes ending at T; treat it as known at T + 5min.
        'available_at': pd.to_datetime(raw['create_time']) + pd.Timedelta(minutes=5),
        'open_interest': pd.to_numeric(raw['sum_open_interest'], errors='coerce'),
        'top_trader_ls': pd.to_numeric(raw['sum_toptrader_long_short_ratio'], errors='coerce'),
        'global_ls': pd.to_numeric(raw['count_long_short_ratio'], errors='coerce'),
        'taker_ls': pd.to_numeric(raw['sum_taker_long_short_vol_ratio'], errors='coerce'),
    })
    return out.dropna(subset=['available_at'])
