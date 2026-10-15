import numpy as np
import pandas as pd
import pytest

from forecast_utils import (
    HORIZON, LOOK_BACK, extract_close, make_windows, next_business_days,
    recursive_forecast, rmse, split_holdout,
)


def prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.normal(0.1, 1, n)),
                     index=pd.bdate_range("2022-01-03", periods=n))


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0
    assert rmse([0, 0], [3, 4]) == pytest.approx((12.5) ** 0.5)
    with pytest.raises(ValueError):
        rmse([1], [1, 2])


def test_holdout_is_the_tail_and_disjoint():
    s = prices()
    train, test = split_holdout(s)
    assert len(test) == HORIZON and len(train) == len(s) - HORIZON
    assert train.index.max() < test.index.min()
    with pytest.raises(ValueError):
        split_holdout(s.iloc[:50])


def test_windows_align_target_with_next_value():
    scaled = np.arange(100, dtype=float).reshape(-1, 1)
    X, y = make_windows(scaled, 10)
    assert X.shape == (90, 10, 1)
    assert X[0, -1, 0] == 9 and y[0] == 10
    assert y[-1] == 99


def test_future_dates_skip_weekends_and_start_after_last_day():
    friday = pd.Timestamp("2024-03-08")
    idx = next_business_days(friday, 5)
    assert idx[0] == pd.Timestamp("2024-03-11")  # Monday
    assert all(d.weekday() < 5 for d in idx)
    assert len(idx) == 5


def test_recursive_forecast_feeds_predictions_back():
    seen = []

    def predict_next(window):
        seen.append(window.copy())
        return window[-1, 0] + 1.0  # "last value + 1"

    out = recursive_forecast(predict_next, np.array([[1.0], [2.0], [3.0]]), 3)
    assert out.ravel().tolist() == [4.0, 5.0, 6.0]
    assert seen[1].ravel().tolist() == [2.0, 3.0, 4.0]


def test_extract_close_handles_flat_and_multiindex_columns():
    s = prices(5)
    flat = pd.DataFrame({"Close": s, "Open": s})
    multi = pd.DataFrame({("Close", "AAPL"): s, ("Open", "AAPL"): s})
    multi.columns = pd.MultiIndex.from_tuples(multi.columns)
    assert extract_close(flat, "AAPL").equals(s.astype(float))
    assert extract_close(multi, "AAPL").equals(s.astype(float))


def test_extract_close_drops_timezone():
    s = prices(5)
    s.index = s.index.tz_localize("America/New_York")
    out = extract_close(pd.DataFrame({"Close": s}), "AAPL")
    assert out.index.tz is None


def test_full_pipeline_smoke():
    pytest.importorskip("tensorflow")
    pytest.importorskip("prophet")
    pytest.importorskip("statsmodels")
    from forecast_utils import run_forecasts
    close = prices(260)
    res = run_forecasts(close, epochs=1)
    assert set(res["rmse"]) == {"LSTM", "ARIMA", "Prophet", "Naive (last train price)"}
    for name in ("LSTM", "ARIMA", "Prophet"):
        assert len(res["holdout"][name]) == HORIZON
        assert len(res["future"][name]) == HORIZON
        assert np.isfinite(res["future"][name]).all()
    assert res["future_index"][0] > close.index[-1]
    assert res["future_index"][0].weekday() < 5
