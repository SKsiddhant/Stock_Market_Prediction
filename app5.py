
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import date

st.title('Stock Price Prediction Comparison')

# Add a dropdown menu for stock selection
selected_stock = st.selectbox(
    'Select a stock ticker:',
    ('AAPL', 'MSFT', 'GOOGL', 'AMZN') # Add more tickers here if needed
)

# Download historical data
today = date.today().strftime("%Y-%m-%d")
data = yf.download(selected_stock, start="2010-01-01", end=today)
if data.empty:
    st.error("No data downloaded. Please check your internet connection or ticker symbol.")
    st.stop()
data.to_csv(f"{selected_stock}_stock.csv")

st.write(f"## Original Data Head for {selected_stock}")
st.write(data.head())


# Data preprocessing
# Handle potential MultiIndex in data
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close'][selected_stock].values.reshape(-1, 1)
else:
    close_prices = data['Close'].values.reshape(-1, 1)

close_prices_series = pd.Series(close_prices.flatten(), index=data.index)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

look_back = 60
X = []
y = []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets for ARIMA and Prophet
train_size = int(len(close_prices_series) * 0.8)
train_data, test_data = close_prices_series[0:train_size], close_prices_series[train_size:]

# Model selection and training
st.write("## Model Training")

with st.spinner('Training LSTM model...'):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=50, batch_size=16, verbose=0)
st.success('LSTM model training complete!')

with st.spinner('Training ARIMA model...'):
    # Using a simple order (5,1,0) as an example, this would typically require tuning
    arima_model = ARIMA(train_data, order=(5, 1, 0))
    arima_model_fit = arima_model.fit()
st.success('ARIMA model training complete!')

with st.spinner('Training Prophet model...'):
    prophet_train_data = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
    prophet_model = Prophet()
    prophet_model.fit(prophet_train_data)
st.success('Prophet model training complete!')


# Model Evaluation (for LSTM on training data as before)
lstm_predicted_train_price = lstm_model.predict(X)
lstm_predicted_train_price = scaler.inverse_transform(lstm_predicted_train_price)
actual_train_price = scaler.inverse_transform(y.reshape(-1, 1))

lstm_rmse = sqrt(mean_squared_error(actual_train_price, lstm_predicted_train_price))
st.write("## Model Evaluation (LSTM on training data)")
st.write(f"Root Mean Squared Error (RMSE): {lstm_rmse}")


# Prediction for the next 30 days
st.write("## Predicting Future Prices (Next 30 Days)")

# Generate ARIMA predictions for the next 30 days
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]
arima_future_forecast = arima_model_fit.forecast(steps=30)
arima_future_prices_series = pd.Series(arima_future_forecast.values, index=future_dates, name='ARIMA Predicted Price')

# Generate Prophet predictions for the next 30 days
prophet_future_dates = pd.DataFrame({'ds': future_dates})
prophet_forecast = prophet_model.predict(prophet_future_dates)
prophet_future_prices_series = pd.Series(prophet_forecast['yhat'].values, index=future_dates, name='Prophet Predicted Price')


# Generate LSTM predictions for the next 30 days
last_look_back_days_scaled = scaled_data[-look_back:]
lstm_predicted_future_prices = []

with st.spinner(f'Predicting next 30 days with LSTM...'):
    for _ in range(30):
        X_test = np.reshape(last_look_back_days_scaled, (1, last_look_back_days_scaled.shape[0], 1))
        predicted_price_scaled = lstm_model.predict(X_test, verbose=0)
        lstm_predicted_future_prices.append(predicted_price_scaled[0, 0])
        last_look_back_days_scaled = np.append(last_look_back_days_scaled[1:], predicted_price_scaled, axis=0)

lstm_predicted_future_prices = scaler.inverse_transform(np.array(lstm_predicted_future_prices).reshape(-1, 1))
lstm_future_prices_series = pd.Series(lstm_predicted_future_prices.flatten(), index=future_dates, name='LSTM Predicted Price')


st.write("Predicted stock prices for the next 30 days:")
predicted_future_df_display = pd.DataFrame({
    'ARIMA': arima_future_prices_series,
    'Prophet': prophet_future_prices_series,
    'LSTM': lstm_future_prices_series
})
st.write(predicted_future_df_display)


# Visualization
st.write("## Stock Price Prediction Plots")

# Plot Actual vs LSTM Predicted
fig_lstm, ax_lstm = plt.subplots(figsize=(14, 7))
ax_lstm.plot(close_prices_series, label='Actual Price')
ax_lstm.plot(lstm_future_prices_series, label='LSTM Predicted Price', linestyle='--')
ax_lstm.set_title(f'{selected_stock} Stock Price Prediction (LSTM)')
ax_lstm.set_xlabel('Date')
ax_lstm.set_ylabel('Price')
ax_lstm.legend()
st.pyplot(fig_lstm)

# Plot Actual vs ARIMA Predicted
fig_arima, ax_arima = plt.subplots(figsize=(14, 7))
ax_arima.plot(close_prices_series, label='Actual Price')
ax_arima.plot(arima_future_prices_series, label='ARIMA Predicted Price', linestyle='--')
ax_arima.set_title(f'{selected_stock} Stock Price Prediction (ARIMA)')
ax_arima.set_xlabel('Date')
ax_arima.set_ylabel('Price')
ax_arima.legend()
st.pyplot(fig_arima)

# Plot Actual vs Prophet Predicted
fig_prophet, ax_prophet = plt.subplots(figsize=(14, 7))
ax_prophet.plot(close_prices_series, label='Actual Price')
ax_prophet.plot(prophet_future_prices_series, label='Prophet Predicted Price', linestyle='--')
ax_prophet.set_title(f'{selected_stock} Stock Price Prediction (Prophet)')
ax_prophet.set_xlabel('Date')
ax_prophet.set_ylabel('Price')
ax_prophet.legend()
st.pyplot(fig_prophet)