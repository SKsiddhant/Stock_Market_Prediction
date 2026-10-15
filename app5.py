import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

from forecast_utils import HORIZON, extract_close, run_forecasts

TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN")

st.title("Stock Price Prediction Comparison")
st.caption(
    "LSTM, ARIMA and Prophet are trained on all but the last %d trading days, scored on those "
    "held-out days, then used to forecast the next %d business days. Educational only - not "
    "financial advice." % (HORIZON, HORIZON)
)

selected_stock = st.selectbox("Select a stock ticker:", TICKERS)
epochs = st.slider("LSTM training epochs", 1, 100, 25,
                   help="More epochs take longer. Training only starts when you press the button.")


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(ticker: str) -> pd.Series:
    data = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    return extract_close(data, ticker)


# Models are expensive: cache per ticker/epochs so a widget change does not retrain them.
@st.cache_resource(show_spinner=False)
def forecasts_for(ticker: str, last_date: str, n_obs: int, epochs: int):
    return run_forecasts(load_prices(ticker), epochs=epochs)


close = load_prices(selected_stock)
if close.empty:
    st.error("No data downloaded. Please check your internet connection or ticker symbol.")
    st.stop()

st.write(f"## Recent prices for {selected_stock}")
st.line_chart(close.tail(500))

if not st.button("Train models and forecast", type="primary"):
    st.info("Press the button to train the three models (the LSTM can take a few minutes).")
    st.stop()

try:
    with st.spinner("Training LSTM, ARIMA and Prophet..."):
        res = forecasts_for(selected_stock, str(close.index[-1].date()), len(close), epochs)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"## Held-out accuracy (last {HORIZON} trading days)")
rmse_df = pd.DataFrame({"RMSE (price units)": res["rmse"]}).sort_values("RMSE (price units)")
st.dataframe(rmse_df)
st.caption("Lower is better. A model that cannot beat the naive 'price stays flat' row has "
           "no predictive value on this window.")

st.write(f"## Forecast: next {HORIZON} business days")
future_df = pd.DataFrame(res["future"], index=res["future_index"])
st.dataframe(future_df)

st.write("## Plots")
for name in ("LSTM", "ARIMA", "Prophet"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(close.tail(250), label="Actual price")
    ax.plot(res["test"].index, res["holdout"][name], linestyle=":", label=f"{name} on held-out days")
    ax.plot(res["future_index"], res["future"][name], linestyle="--", label=f"{name} forecast")
    ax.set_title(f"{selected_stock} - {name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
