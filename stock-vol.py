import yfinance as yf
import pandas as pd

stock = yf.Ticker("TSLA")
volatility_data = stock.history(period="max")

#exploratory data analysis to understand and spot anomalies
import matplotlib.pyplot as plot

# Plot historical data
plot.figure(figsize=(8, 4))
plot.plot(volatility_data.index, volatility_data["Close"])
plot.xlabel("Date")
plot.ylabel("Volatility")
plot.title("Historical Data")
plot.grid(True)

plot.show()

# rolling mean and rolling std assist in identifying times of increased volatility (e.g. higher std -> higher volatility)
rolling_mean = volatility_data["Close"].rolling(window=30).mean()
rolling_std = volatility_data["Close"].rolling(window=30).std()

# plot the rolling mean and standard deviation
plot.figure(figsize=(8, 4))
plot.plot(volatility_data.index, volatility_data["Close"], label="Volatility")
plot.plot(rolling_mean.index, rolling_mean, label="Rolling Mean")
plot.plot(rolling_std.index, rolling_std, label="Rolling Std")
plot.xlabel("Date")
plot.ylabel("Volatility")
plot.title("Rolling Mean and Standard Deviation of Volatility")
plot.legend()
plot.grid(True)

plot.show()

import numpy as np
from arch import arch_model

returns = np.log(volatility_data["Close"]).diff().dropna()
#Garch(1,1)
model = arch_model(returns, vol="Garch", p=1, q=1)
results = model.fit()

# Estimate the volatility (conditional variance?)
#plots conditional variance and log returns
volatility = results.conditional_volatility
plot.figure(figsize=(8, 4))
plot.plot(volatility.index, volatility, label="Estimated Volatility")
plot.plot(returns.index, returns, label="Actual Volatility")
plot.xlabel("Date")
plot.ylabel("Volatility")
plot.title("Estimated and Actual Volatility")
plot.legend()
plot.grid(True)

plot.show()
#estimated volatility is much higher than actual volatility (fixed cuz i scaled wrong oops)

# Forecasting
forecast = results.forecast(start=0, horizon=30)
forecast_volatility = forecast.variance.dropna().values.flatten()

# Plot forecasted volatility
plot.figure(figsize=(8, 4))
plot.plot(forecast_volatility, label="Forecasted Volatility")
plot.xlabel("Time")
plot.ylabel("Volatility")
plot.title("Forecasted Volatility")
plot.legend()
plot.grid(True)

plot.show()

# Evaluate model performance
# Mean Absolute Error (MAE)
mae = np.mean(np.abs(volatility - returns))
print("Mean Absolute Error (MAE):", mae)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((volatility - returns) ** 2))
print("Root Mean Squared Error (RMSE):", rmse)

forecast_errors = volatility - returns
plot.figure(figsize=(8, 4))
plot.hist(forecast_errors, bins=30, density=True)
plot.xlabel("Forecast Error")
plot.ylabel("Density")
plot.title("Histogram of Forecast Errors")
plot.grid(True)

plot.show()