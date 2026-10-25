import json
import time

import numpy as np
import pytest

from backend.models import TORCH_AVAILABLE
from backend.prediction_service import PredictionService, AnalysisCancelled
from tests.conftest import FakeFetcher


def test_analyze_end_to_end():
    service = PredictionService(fetcher=FakeFetcher())
    result = service.analyze('BTC', ['1h', '1d'], ['gbm', 'logreg'])
    json.dumps(result)  # must be JSON-serialisable for the API
    assert set(result['timeframes']) == {'1h', '1d'} and not result['errors']
    tf = result['timeframes']['1h']
    sig = tf['signal']
    assert sig['direction'] in ('up', 'down', 'neutral')
    assert abs(sum(sig['probabilities'].values()) - 1) < 1e-3
    assert sig['confidence_level'] in ('low', 'medium', 'high')
    for name in ('gbm', 'logreg', 'ensemble', 'baseline_majority', 'baseline_persistence'):
        assert 0 <= tf['metrics'][name]['accuracy'] <= 1
    assert tf['backtest']['equity_curve']['time']
    assert tf['feature_importance']
    # AR(1) synthetic data has real momentum, so a working model calls direction better than chance
    # (53-57% across seeds; a leaking or broken pipeline would be far off).
    assert tf['metrics']['gbm']['directional_accuracy'] > 0.52
    assert 'gbm_p_up' in service.export_csv('BTC', '1h').splitlines()[0]


def test_altcoin_uses_btc_context():
    fetcher = FakeFetcher()
    service = PredictionService(fetcher=fetcher)
    result = service.analyze('ETH', ['4h'], ['logreg'])
    assert ('BTC', '4h') in fetcher.calls
    assert result['timeframes']['4h']['data']['features'] > 40


def test_signal_after_reload_from_disk():
    PredictionService(fetcher=FakeFetcher()).analyze('BTC', ['4h'], ['gbm', 'logreg'])
    fresh = PredictionService(fetcher=FakeFetcher())  # models must come from disk
    out = fresh.latest_signal('BTC', ['4h'])
    assert not out['errors'] and out['signals']['4h']['direction'] in ('up', 'down', 'neutral')


def test_signal_without_training_reports_error():
    out = PredictionService(fetcher=FakeFetcher()).latest_signal('BTC', ['1d'])
    assert '1d' in out['errors']


def test_binary_mode(isolated_config):
    isolated_config.app_config.label_mode = 'binary'
    result = PredictionService(fetcher=FakeFetcher()).analyze('BTC', ['1d'], ['logreg'])
    assert set(result['timeframes']['1d']['signal']['probabilities']) == {'down', 'up'}


def test_cancel():
    import threading
    event = threading.Event()
    event.set()
    with pytest.raises(AnalysisCancelled):
        PredictionService(fetcher=FakeFetcher()).analyze('BTC', ['1h'], ['logreg'], cancel_event=event)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize('name', ['gru', 'lstm', 'transformer'])
def test_sequence_models(name):
    service = PredictionService(fetcher=FakeFetcher(n=900))
    result = service.analyze('BTC', ['1d'], [name])
    assert result['timeframes']['1d']['metrics'][name]['n'] > 0
    reloaded = PredictionService(fetcher=FakeFetcher(n=900)).latest_signal('BTC', ['1d'])
    assert not reloaded['errors']


def test_api_flow(monkeypatch):
    from backend import api
    monkeypatch.setattr(api, 'prediction_service', PredictionService(fetcher=FakeFetcher()))
    client = api.app.test_client()

    assert client.get('/api/models').status_code == 200
    assert len(client.get('/api/timeframes').get_json()['timeframes']) == 3
    assert client.post('/api/analyze', json={'symbol': 'BTC', 'timeframes': ['5m']}).status_code == 400

    resp = client.post('/api/analyze', json={'symbol': 'btc', 'timeframes': ['1d'], 'models': ['logreg']})
    assert resp.status_code == 202
    task_id = resp.get_json()['task_id']
    for _ in range(300):
        status = client.get(f'/api/analyze/{task_id}').get_json()
        if status['status'] in ('completed', 'failed'):
            break
        time.sleep(0.1)
    assert status['status'] == 'completed', status.get('error')
    assert status['result']['timeframes']['1d']['signal']['direction']

    assert client.get('/api/signal/BTC?timeframes=1d').get_json()['signals']['1d']
    csv = client.get('/api/export/BTC/1d')
    assert csv.status_code == 200 and csv.mimetype == 'text/csv'
    assert client.get('/api/results/BTC/4h').status_code == 404
