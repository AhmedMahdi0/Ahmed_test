"""fetch_futures against a fake Binance (formats as documented for the public endpoints)."""
import io
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

from backend import data_fetcher
from backend.data_fetcher import CryptoDataFetcher

HOUR = 3600 * 1000


class FakeResponse:
    def __init__(self, payload=None, status=200, content=b''):
        self.payload, self.status_code, self.content = payload, status, content

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')


class FakeFapi:
    """Serves funding (paginated at 1000), premium klines (paginated at 1500) and the 30-day stats."""

    def __init__(self, now_ms):
        self.now = now_ms
        self.calls = []
        self.funding_times = list(range(now_ms - 2500 * 8 * HOUR, now_ms, 8 * HOUR))

    def get(self, url, params=None, timeout=None):
        path = url.replace(data_fetcher.FAPI_HOST, '')
        self.calls.append(path)
        start = params.get('startTime', 0)
        if path == '/fapi/v1/fundingRate':
            rows = [t for t in self.funding_times if t >= start][:params['limit']]
            return FakeResponse([{'symbol': 'BTCUSDT', 'fundingTime': t, 'fundingRate': '0.00010000',
                                  'markPrice': '50000'} for t in rows])
        if path == '/fapi/v1/premiumIndexKlines':
            first = max(start, self.now - 3000 * HOUR) // HOUR * HOUR
            opens = list(range(first, self.now, HOUR))[:params['limit']]
            return FakeResponse([[t, '0', '0', '0', '0.0002', '0', t + HOUR - 1, '0', 0, '0', '0', '0']
                                 for t in opens])
        field = {'/futures/data/openInterestHist': 'sumOpenInterest',
                 '/futures/data/topLongShortPositionRatio': 'longShortRatio',
                 '/futures/data/globalLongShortAccountRatio': 'longShortRatio',
                 '/futures/data/takerlongshortRatio': 'buySellRatio'}[path]
        stamps = list(range(self.now // HOUR * HOUR - 499 * HOUR, self.now, HOUR))
        return FakeResponse([{'symbol': 'BTCUSDT', field: '1.5', 'timestamp': t} for t in stamps])


def _zip_csv(day):
    times = pd.date_range(day, periods=288, freq='5min') + pd.Timedelta(minutes=5)
    lines = ['create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,'
             'sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio']
    lines += [f'{t:%Y-%m-%d %H:%M:%S},BTCUSDT,80000,4e9,1.5,1.2,1.8,1.0' for t in times]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('metrics.csv', '\n'.join(lines))
    return buf.getvalue()


def test_fetch_futures_parses_and_paginates(monkeypatch, tmp_path):
    now = int(time.time() * 1000)
    fapi = FakeFapi(now)
    downloads = []

    def fake_vision_get(url, timeout=None):
        downloads.append(url)
        day = url.rsplit('-metrics-', 1)[1].replace('.zip', '')
        if day < '2024-01-01':
            return FakeResponse(status=404)
        return FakeResponse(content=_zip_csv(day))

    monkeypatch.setattr(data_fetcher.requests, 'get', fake_vision_get)
    fetcher = CryptoDataFetcher(session=fapi)
    out = fetcher.fetch_futures('BTC', '1h', history_days=1000)

    funding = out['funding']
    assert fapi.calls.count('/fapi/v1/fundingRate') >= 3                 # 1000 per page
    assert funding.index.is_monotonic_increasing and not funding.index.duplicated().any()
    assert np.isclose(funding['funding_rate'].iloc[0], 1e-4)
    assert len(out['premium']) > 1500 and np.isclose(out['premium']['premium'].iloc[-1], 0.0002)
    m = out['metrics']
    assert set(m.columns) == set(data_fetcher.METRIC_COLUMNS)
    assert m.index.max() > pd.Timestamp.now() - pd.Timedelta(hours=3)    # recent API fills the archive lag
    assert not out['warnings']

    # The archive is cached on disk: a second fetcher downloads nothing new.
    n = len(downloads)
    CryptoDataFetcher(session=FakeFapi(now)).fetch_futures('BTC', '1h', history_days=1000)
    assert len(downloads) == n


def test_fetch_futures_without_any_data_raises(monkeypatch):
    class Down:
        def get(self, *a, **k):
            return FakeResponse(status=451)

    monkeypatch.setattr(data_fetcher.requests, 'get', lambda *a, **k: FakeResponse(status=451))
    with pytest.raises(data_fetcher.DataFetchError):
        CryptoDataFetcher(session=Down()).fetch_futures('BTC', '4h', history_days=30)
