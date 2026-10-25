import numpy as np
import pandas as pd

from backend.config import get_config
from backend.evaluation import walk_forward_splits, split_train_val
from backend.feature_engineering import FeatureEngineer, classify_returns
from tests.conftest import make_candles, FakeFetcher


def test_features_do_not_look_ahead():
    """Features on a row must not change when later candles are added or altered."""
    fe = FeatureEngineer()
    full = make_candles(800, '4h')
    cut = 600
    altered = full.copy()
    altered.iloc[cut:, :4] *= 3.0  # wildly different future
    a = fe.make_dataset(full, '4h').X
    b = fe.make_dataset(altered, '4h').X
    common = a.index[a.index < full.index[cut]]
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_label_is_next_candle_direction():
    fe = FeatureEngineer()
    df = make_candles(600, '1h')
    ds = fe.make_dataset(df, '1h')
    thr = ds.threshold
    close = df['close']
    for t in ds.X.index[:-1][::37]:
        i = close.index.get_loc(t)
        r = np.log(close.iloc[i + 1] / close.iloc[i])
        expected = 2 if r > thr else (0 if r < -thr else 1)
        assert ds.y.loc[t] == expected
    assert np.isnan(ds.y.iloc[-1])  # the future of the last candle is unknown


def test_binary_labels():
    r = pd.Series([0.02, -0.01, 0.0, np.nan])
    out = classify_returns(r, 0.01, 'binary')
    assert out.iloc[:3].tolist() == [1, 0, 0] and np.isnan(out.iloc[3])


def test_macro_values_only_after_the_day_is_over():
    fe = FeatureEngineer()
    df = make_candles(24 * 10, '1h', start='2023-03-01')
    days = pd.date_range('2023-02-20', '2023-03-15', freq='D')
    macro = pd.DataFrame({'FGI': np.arange(len(days), dtype=float)}, index=days)
    f = fe.macro_features(macro, df.index, '1h')
    # Candle 2023-03-05 10:00 closes at 11:00; the latest finished day is 2023-03-04.
    expected = macro.loc['2023-03-04', 'FGI'] / 100
    assert np.isclose(f.loc['2023-03-05 10:00', 'fgi'], expected)
    # The 23:00 candle closes exactly at midnight, when 2023-03-05 is complete.
    assert np.isclose(f.loc['2023-03-05 23:00', 'fgi'], macro.loc['2023-03-05', 'FGI'] / 100)


def test_walk_forward_is_chronological_and_purged():
    splits = walk_forward_splits(1000, 4, 0.6, purge=3)
    assert len(splits) == 4
    for train, test in splits:
        assert train.max() + 3 < test.min() + 1
        assert train.max() < test.min() - 2
    tests = np.concatenate([t for _, t in splits])
    assert (np.diff(tests) == 1).all() and tests[-1] == 999
    fit, val = split_train_val(np.arange(500), 0.2, purge=2)
    assert fit.max() + 2 < val.min()
