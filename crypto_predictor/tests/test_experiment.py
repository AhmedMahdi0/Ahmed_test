import numpy as np

from experiments.compare_features import main, block_bootstrap_ci
from tests.conftest import make_candles


def test_block_bootstrap_detects_a_real_shift():
    rng = np.random.default_rng(0)
    lo, hi, p = block_bootstrap_ci(rng.normal(0.5, 1, 2000), block=20)
    assert lo > 0 and p < 0.01
    lo, hi, p = block_bootstrap_ci(rng.normal(0.0, 1, 2000), block=20)
    assert lo < 0 < hi


def test_compare_features_runs_on_csv(tmp_path):
    csv = tmp_path / 'candles.csv'
    make_candles(3000, '1h').to_csv(csv)
    report = main(['--experiment', 'htf', '--csv', str(csv), '--csv-timeframe', '1h', '--timeframes', '1h', '4h',
                   '--models', 'logreg', '--folds', '2', '--out', str(tmp_path / 'r.json')])
    for tf in ('1h', '4h'):
        cmp = report['timeframes'][tf]
        assert cmp['verdict'] in ('IMPROVED', 'WORSE', 'NO SIGNIFICANT DIFFERENCE')
        assert cmp['paired_candles'] > 100
    assert (tmp_path / 'r.json').exists()
