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
    assert sig['direction'] in ('up', 'down', 'none')
    assert abs(sum(sig['probabilities'].values()) - 1) < 1e-3
    assert sig['confidence_level'] in ('low', 'medium', 'high', None)
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
    assert not out['errors'] and out['signals']['4h']['direction'] in ('up', 'down', 'none')


def test_signal_without_training_reports_error():
    out = PredictionService(fetcher=FakeFetcher()).latest_signal('BTC', ['1d'])
    assert '1d' in out['errors']


def test_ternary_mode(isolated_config):
    isolated_config.app_config.label_mode = 'ternary'
    result = PredictionService(fetcher=FakeFetcher()).analyze('BTC', ['1d'], ['logreg'])
    tf = result['timeframes']['1d']
    assert set(tf['signal']['probabilities']) == {'down', 'neutral', 'up'}
    assert tf['training']['drop_weak_moves'] is False   # weak moves are the neutral class here


@pytest.mark.parametrize('keep', [0.2, 0.6])
def test_signal_filter_keeps_about_the_requested_share(isolated_config, keep):
    isolated_config.app_config.signal_keep_fraction = keep
    result = PredictionService(fetcher=FakeFetcher(n=3000)).analyze('BTC', ['1h'], ['gbm', 'logreg'])
    tf = result['timeframes']['1h']
    sig = tf['metrics']['ensemble']['signals']
    # The cutoff is set on validation data, so test coverage is close to, not exactly, `keep`.
    assert abs(sig['coverage'] - keep) < 0.15
    shown = [s for s in tf['history']['signal'] if s != 'none']
    assert shown and set(shown) <= {'up', 'down'}
    assert tf['signal_filter']['enabled'] and len(tf['signal_filter']['fold_cutoffs']) == 3


def test_signal_filter_off_shows_every_directional_call(isolated_config):
    isolated_config.app_config.signal_filter = False
    tf = PredictionService(fetcher=FakeFetcher()).analyze('BTC', ['4h'], ['logreg'])['timeframes']['4h']
    assert tf['metrics']['ensemble']['signals']['coverage'] == 1.0   # binary: every call is up or down
    assert tf['signal']['direction'] in ('up', 'down')


def test_weak_moves_are_left_out_of_training(isolated_config, monkeypatch):
    from backend import prediction_service as ps
    seen = []
    real_create = ps.create_model

    def spy(name, n_classes, tf):
        model = real_create(name, n_classes, tf)
        real_fit = model.fit

        def fit(X, y, fit_idx, val_idx=None, cancel_check=None):
            seen.append(fit_idx)
            return real_fit(X, y, fit_idx, val_idx, cancel_check)
        model.fit = fit
        return model

    monkeypatch.setattr(ps, 'create_model', spy)
    service = PredictionService(fetcher=FakeFetcher())
    ds = service._dataset('BTC', '1h')
    service.analyze('BTC', ['1h'], ['logreg'])
    strong = np.abs(np.nan_to_num(ds.fwd_return.to_numpy())) > ds.threshold
    assert seen and all(strong[idx].all() for idx in seen)

    isolated_config.app_config.drop_weak_moves = False
    seen.clear()
    service.analyze('BTC', ['1h'], ['logreg'])
    assert any(not strong[idx].all() for idx in seen)


def test_weak_live_signal_is_hidden():
    service = PredictionService(fetcher=FakeFetcher())
    service.analyze('BTC', ['4h'], ['logreg'])
    bundle = service._get_bundle('BTC', '4h')
    ds = service._dataset('BTC', '4h')
    X = ds.X[bundle['meta']['feature_columns']].to_numpy(dtype=np.float32)
    sig = service._signal(ds, X, bundle['models'], None, cutoff=0.999)
    assert sig['direction'] == 'none' and sig['filtered'] and sig['confidence_level'] is None
    assert sig['raw_direction'] in ('up', 'down')


def test_futures_and_htf_groups_in_service(isolated_config):
    isolated_config.app_config.use_futures = True
    isolated_config.app_config.use_htf_context = True
    tf = PredictionService(fetcher=FakeFetcher(n=3000)).analyze('ETH', ['1h'], ['gbm'])['timeframes']['1h']
    groups = tf['data']['feature_groups']
    assert groups['futures'] >= 10 and groups['higher_timeframe'] >= 18


def test_missing_futures_data_is_reported_not_fatal(isolated_config):
    isolated_config.app_config.use_futures = True

    class NoFutures(FakeFetcher):
        def fetch_futures(self, symbol, timeframe, history_days=None):
            raise RuntimeError('blocked')

    tf = PredictionService(fetcher=NoFutures()).analyze('BTC', ['4h'], ['logreg'])['timeframes']['4h']
    assert tf['data']['feature_groups']['futures'] == 0
    assert any('futures' in note for note in tf['data']['notes'])


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
