import numpy as np
import pandas as pd
import pytest

from backend.config import get_config
from backend.evaluation import walk_forward_splits, split_train_val
from backend.feature_engineering import FeatureEngineer, classify_returns
from tests.conftest import make_candles, make_futures, FakeFetcher


@pytest.mark.parametrize('support_resistance', [False, True])
def test_features_do_not_look_ahead(support_resistance):
    """Features on a row must not change when later candles are added or altered."""
    get_config().app_config.use_support_resistance = support_resistance
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
    get_config().app_config.label_mode = 'ternary'
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


def test_support_resistance_levels_appear_only_after_confirmation():
    cfg = get_config().app_config
    cfg.sr_pivot_bars, cfg.atr_window = 2, 3
    # A "tent": highs rise to a single peak at index 10, then fall. Only one swing point exists.
    n = 30
    high = 110.0 - 0.5 * np.abs(np.arange(n) - 10)
    close = high - 1.0
    df = pd.DataFrame({'open': close, 'high': high, 'low': high - 2.0, 'close': close, 'volume': 1.0,
                       'quote_volume': 1.0, 'trades': 1.0, 'taker_buy_base': 0.5},
                      index=pd.date_range('2024-01-01', periods=n, freq='1h'))
    f = FeatureEngineer().support_resistance_features(df)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(3).mean()
    # The swing high at 10 needs 2 candles on its right: unknown at 11, a level from 12 on.
    assert f['sr_res_dist_atr'].iloc[11] == 10.0  # capped: "no resistance in reach"
    for t in (12, 20):
        assert np.isclose(f['sr_res_dist_atr'].iloc[t], min((110 - close[t]) / atr.iloc[t], 10.0))
        assert f['sr_res_strength'].iloc[t] == 1
    assert f['sr_sup_dist_atr'].iloc[20] == 10.0  # no swing low anywhere -> no support


def test_support_resistance_features_are_optional():
    df = make_candles(1000, '4h')
    off = FeatureEngineer().make_dataset(df, '4h').X
    get_config().app_config.use_support_resistance = True
    on = FeatureEngineer().make_dataset(df, '4h').X
    assert not any(c.startswith('sr_') for c in off.columns)
    assert {'sr_res_dist_atr', 'sr_sup_dist_atr', 'sr_position', 'sr_break_up'} <= set(on.columns)
    assert len(on) == len(off)  # same rows, so with/without runs are directly comparable


def _with_groups(df, futures):
    from backend.feature_engineering import resample_ohlcv
    higher = {'4h': resample_ohlcv(df, '4h'), '1d': resample_ohlcv(df, '1d')}
    return FeatureEngineer().make_dataset(df, '1h', higher=higher, futures=futures).X


def test_htf_and_futures_features_do_not_look_ahead():
    cfg = get_config().app_config
    cfg.use_htf_context = cfg.use_futures = True
    full = make_candles(2500, '1h')
    fut = make_futures(full)
    cut_time = full.index[1900] + pd.Timedelta(minutes=17)   # not on any 4h/1d boundary
    altered = full.copy()
    altered.loc[altered.index >= cut_time, ['open', 'high', 'low', 'close']] *= 2.5
    fut_alt = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in fut.items()}
    for key in ('funding', 'premium', 'metrics'):
        frame = fut_alt[key]
        frame.loc[frame.index >= cut_time] *= 3.0
    a, b = _with_groups(full, fut), _with_groups(altered, fut_alt)
    assert any(c.startswith('htf_1d_') for c in a.columns) and any(c.startswith('fut_oi') for c in a.columns)
    # A 1h candle opening at T closes at T+1h, so it may use data up to T+1h.
    common = a.index[a.index + pd.Timedelta(hours=1) <= cut_time]
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_htf_uses_last_closed_higher_candle():
    from backend.feature_engineering import resample_ohlcv
    df = make_candles(24 * 40, '1h')
    daily = resample_ohlcv(df, '1d')
    f = FeatureEngineer().htf_features(df, '1h', {'1d': daily})
    ret_daily = np.log(daily['close']).diff()
    # 1h candle 10:00 on day D closes at 11:00: the latest closed daily candle is D-1.
    t = pd.Timestamp(df.index[24 * 30].normalize()) + pd.Timedelta(hours=10)
    assert np.isclose(f.loc[t, 'htf_1d_ret_1'], ret_daily.loc[t.normalize() - pd.Timedelta(days=1)])
    # The 23:00 candle closes at midnight, when day D itself is complete.
    t_last = t.normalize() + pd.Timedelta(hours=23)
    assert np.isclose(f.loc[t_last, 'htf_1d_ret_1'], ret_daily.loc[t.normalize()])
    # Forming-candle return at 10:00: from the day's open to the 10:00 candle's close.
    assert np.isclose(f.loc[t, 'htf_1d_partial_ret'], np.log(df.loc[t, 'close'] / df.loc[t.normalize(), 'open']))


def test_futures_values_align_to_when_they_were_known():
    df = make_candles(500, '1h', start='2024-01-01')
    fut = make_futures(df)
    f = FeatureEngineer().futures_features(fut, df, '1h')
    t = pd.Timestamp('2024-01-05 07:00')            # closes 08:00, exactly a funding settlement
    assert np.isclose(f.loc[t, 'fut_funding'], fut['funding'].loc['2024-01-05 08:00', 'funding_rate'] * 1e4)
    t2 = pd.Timestamp('2024-01-05 06:00')           # closes 07:00 -> still the 00:00 settlement
    assert np.isclose(f.loc[t2, 'fut_funding'], fut['funding'].loc['2024-01-05 00:00', 'funding_rate'] * 1e4)
    oi = fut['metrics']['open_interest']
    expected = np.log(oi.loc['2024-01-05 08:00'] / oi.loc['2024-01-05 07:00'])
    assert np.isclose(f.loc[t, 'fut_oi_chg_1'], expected)


def test_stale_futures_drop_latest_rows_and_say_so():
    cfg = get_config().app_config
    cfg.use_futures = True
    df = make_candles(1500, '1h')
    fut = make_futures(df)
    fut['metrics'] = fut['metrics'][fut['metrics'].index < df.index[-1] - pd.Timedelta(hours=12)]
    ds = FeatureEngineer().make_dataset(df, '1h', futures=fut)
    assert ds.X.index[-1] < df.index[-1]
    assert any('lacks some features' in n for n in ds.notes)


def test_parse_vision_metrics_with_and_without_header():
    from backend.data_fetcher import parse_vision_metrics
    header = ('create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,'
              'sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio')
    rows = ['2024-01-01 00:05:00,BTCUSDT,80000.5,3.4e9,1.6,1.2,1.9,0.95',
            '2024-01-01 00:10:00,BTCUSDT,80010.0,3.4e9,1.6,1.21,1.9,1.05']
    import io
    with_header = parse_vision_metrics(pd.read_csv(io.StringIO('\n'.join([header] + rows))))
    without = parse_vision_metrics(pd.read_csv(io.StringIO('\n'.join(rows))))
    for df in (with_header, without):
        assert len(df) == 2
        assert df['available_at'].iloc[0] == pd.Timestamp('2024-01-01 00:10:00')   # +5 min
        assert np.isclose(df['open_interest'].iloc[1], 80010.0) and np.isclose(df['taker_ls'].iloc[1], 1.05)
