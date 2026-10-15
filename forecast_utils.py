"""Forecasting helpers for the stock prediction app.

Heavy libraries (tensorflow, statsmodels, prophet) are imported lazily inside the
functions that need them, so the pure helpers can be tested without installing them.
"""
from math import sqrt

import numpy as np
import pandas as pd

LOOK_BACK = 60
HORIZON = 30


def rmse(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same length")
    return float(sqrt(np.mean((actual - predicted) ** 2)))


def split_holdout(series: pd.Series, horizon: int = HORIZON):
    """Last `horizon` observations are held out; the models never see them while training."""
    if len(series) <= LOOK_BACK + horizon + 10:
        raise ValueError(
            f"Need more than {LOOK_BACK + horizon + 10} price points, got {len(series)}."
        )
    return series.iloc[:-horizon], series.iloc[-horizon:]


def make_windows(scaled: np.ndarray, look_back: int = LOOK_BACK):
    """Sliding windows: X[i] = scaled[i:i+look_back], y[i] = scaled[i+look_back]."""
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    return X.reshape((X.shape[0], X.shape[1], 1)), np.array(y)


def next_business_days(last_date, n: int) -> pd.DatetimeIndex:
    """The next `n` weekdays after `last_date` (markets are closed on weekends)."""
    return pd.bdate_range(start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=n)


def extract_close(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Close prices as a float Series, whether yfinance returned flat or MultiIndex columns."""
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None) if close.index.tz is not None \
        else pd.to_datetime(close.index)
    return close


def recursive_forecast(predict_next, scaled_tail: np.ndarray, steps: int) -> np.ndarray:
    """Feed each prediction back in as the newest input. `predict_next(window)->float`."""
    window = scaled_tail.copy()
    out = []
    for _ in range(steps):
        nxt = float(predict_next(window))
        out.append(nxt)
        window = np.append(window[1:], [[nxt]], axis=0)
    return np.array(out).reshape(-1, 1)


def build_lstm(look_back: int = LOOK_BACK):
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50), Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def run_forecasts(close: pd.Series, horizon: int = HORIZON, epochs: int = 50, seed: int = 42):
    """Train LSTM, ARIMA and Prophet on everything except the last `horizon` days.

    Returns a dict with, for each model, the forecast over the held-out days (for an
    honest out-of-sample RMSE) and the forecast for the `horizon` business days after
    the last observation. Scaling is fitted on the training part only.
    """
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.tsa.arima.model import ARIMA

    train, test = split_holdout(close, horizon)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
    scaled_all = scaler.transform(close.values.reshape(-1, 1))  # no refit: no leakage

    # ---- LSTM ----
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    X, y = make_windows(scaled_train)
    lstm = build_lstm()
    lstm.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

    def step(window):
        return lstm.predict(window.reshape(1, LOOK_BACK, 1), verbose=0)[0, 0]

    hold_scaled = recursive_forecast(step, scaled_train[-LOOK_BACK:], horizon)
    fut_scaled = recursive_forecast(step, scaled_all[-LOOK_BACK:], horizon)
    lstm_hold = scaler.inverse_transform(hold_scaled).ravel()
    lstm_future = scaler.inverse_transform(fut_scaled).ravel()

    # ---- ARIMA ----
    # Positional integer index: business-day gaps would otherwise trip statsmodels' frequency checks.
    n_train = len(train)
    arima_fit = ARIMA(pd.Series(train.values, index=pd.RangeIndex(n_train)), order=(5, 1, 0)).fit()
    arima_hold = np.asarray(arima_fit.forecast(steps=horizon))
    # extend the fitted model with the held-out days (no refit) before forecasting forward
    held_out = pd.Series(test.values, index=pd.RangeIndex(n_train, n_train + len(test)))
    arima_future = np.asarray(arima_fit.append(held_out, refit=False).forecast(steps=horizon))

    # ---- Prophet ----
    from prophet import Prophet
    prophet = Prophet()
    prophet.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    future_idx = next_business_days(close.index[-1], horizon)
    prophet_hold = prophet.predict(pd.DataFrame({"ds": test.index}))["yhat"].values
    prophet_future = prophet.predict(pd.DataFrame({"ds": future_idx}))["yhat"].values

    return {
        "test": test,
        "future_index": future_idx,
        "holdout": {"LSTM": lstm_hold, "ARIMA": arima_hold, "Prophet": prophet_hold},
        "future": {"LSTM": lstm_future, "ARIMA": arima_future, "Prophet": prophet_future},
        "rmse": {
            "LSTM": rmse(test.values, lstm_hold),
            "ARIMA": rmse(test.values, arima_hold),
            "Prophet": rmse(test.values, prophet_hold),
            "Naive (last train price)": rmse(test.values, np.full(len(test), train.iloc[-1])),
        },
    }
