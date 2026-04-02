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