# Stock Price Prediction Comparison

A Streamlit app that downloads daily prices from Yahoo Finance and compares three forecasting models - **LSTM**, **ARIMA(5,1,0)** and **Prophet** - for AAPL, MSFT, GOOGL and AMZN.

## How it evaluates

1. The last 30 trading days are **held out**; the models never see them while training (the LSTM's scaler is fitted on the training part only).
2. Each model forecasts those 30 days and is scored by RMSE against what really happened, next to a **naive "price stays flat"** baseline. A model that cannot beat the baseline has no predictive value on that window.
3. The same models then forecast the **next 30 business days** (weekends skipped). ARIMA is extended with the held-out days before forecasting; the LSTM uses the latest 60 days as its input window.

Stock prices are close to a random walk, so expect the baseline to be hard to beat. This is a learning project, not trading advice.

## Run

```bash
pip install -r requirements.txt
streamlit run app5.py
```

Training only starts when you press **Train models and forecast** (the LSTM can take a few minutes; lower the epochs slider for a quick try). Results are cached per ticker/epoch setting.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Pure helpers (windowing, hold-out split, business-day dates, RMSE) are in `forecast_utils.py` and are tested without any network. The pipeline and app tests run the real LSTM/ARIMA/Prophet on synthetic prices with one epoch; they are skipped automatically if TensorFlow/Prophet are not installed.

The old `AAPL_stock.csv` snapshot was removed from the repo (375 KB, never read by the app, which always downloads fresh prices); it is still in the history on `main`.
