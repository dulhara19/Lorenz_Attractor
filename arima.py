import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (Example: Air Passenger Data)
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", parse_dates=['Month'], index_col='Month')

# Plot the data
plt.figure(figsize=(10,5))
plt.plot(df, label="Number of Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.title("Airline Passengers Over Time")
plt.legend()
plt.show()


import numpy as np
from statsmodels.tsa.stattools import adfuller

# Calculate rolling mean and standard deviation
rolling_mean = df['Passengers'].rolling(window=12).mean()
rolling_std = df['Passengers'].rolling(window=12).std()

# Plot original data with rolling stats
plt.figure(figsize=(10,5))
plt.plot(df, label="Original Data")
plt.plot(rolling_mean, label="Rolling Mean", color='red')
plt.plot(rolling_std, label="Rolling Std Dev", color='green')
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.title("Rolling Mean & Std Dev")
plt.legend()
plt.show()

# Perform ADF test
adf_test = adfuller(df['Passengers'])
print("ADF Test Statistic:", adf_test[0])
print("p-value:", adf_test[1]) # p-value = 0.9 indicates not stationarity so let's difference the data
# First order differencing
df['Differenced'] = df['Passengers'].diff()

# Drop the first NaN value
df = df.dropna()

# Plot differenced data
plt.figure(figsize=(10,5))
plt.plot(df['Differenced'], label="First Order Differencing", color='purple')
plt.xlabel("Year")
plt.ylabel("Passengers Difference")
plt.title("First Order Differencing")
plt.legend()
plt.show()

adf_test_diff = adfuller(df['Differenced'])
print("ADF Test Statistic:", adf_test_diff[0])
print("p-value:", adf_test_diff[1])
# p-value = 0.05 indicates stationarity lets move to second difference
# Second order differencing

df['Second_Differenced'] = df['Differenced'].diff()
df = df.dropna()  # Drop NaN values

# Plot second differenced data
plt.figure(figsize=(10,5))
plt.plot(df['Second_Differenced'], label="Second Order Differencing", color='orange')
plt.xlabel("Year")
plt.ylabel("Passengers Difference")
plt.title("Second Order Differencing")
plt.legend()
plt.show()

adf_test_2nd_diff = adfuller(df['Second_Differenced'])
print("ADF Test Statistic:", adf_test_2nd_diff[0])
print("p-value:", adf_test_2nd_diff[1])

#Step 5: Finding ARIMA Parameters (p, d, q)
"""
Now that we have a stationary dataset, we need to determine the values of p, d, q for the ARIMA model:

    p (AutoRegressive order): Number of past values to consider (from PACF plot)

    d (Differencing order): Number of times we differenced (we used d = 2)

    q (Moving Average order): Number of past error terms to consider (from ACF plot)
    Plot ACF and PACF to Find p and q

To determine p and q, we use Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.
"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF Plot (for q)
plt.figure(figsize=(10,5))
plot_acf(df['Second_Differenced'], lags=20)
plt.show()

# PACF Plot (for p)
plt.figure(figsize=(10,5))
plot_pacf(df['Second_Differenced'], lags=20)
plt.show()

from statsmodels.tsa.stattools import acf, pacf

# Compute ACF and PACF values
acf_values = acf(df['Second_Differenced'], nlags=20)
pacf_values = pacf(df['Second_Differenced'], nlags=20)

# Find the first lag where ACF values drop near zero (q)
q = next((i for i, val in enumerate(acf_values) if abs(val) < 0.2), None)

# Find the first lag where PACF values drop near zero (p)
p = next((i for i, val in enumerate(pacf_values) if abs(val) < 0.2), None)

print(f"Suggested values: p = {p}, q = {q}")

# now lets build the model

from statsmodels.tsa.arima.model import ARIMA

# Build the ARIMA model
model = ARIMA(df['Passengers'], order=(1, 2, 1))

# Fit the model
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())

# Plot residuals
model_fit.resid.plot(kind='kde')
plt.title("Residuals")
plt.show()

# Forecast next 12 steps (months, years, etc.)
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)

# Print the forecasted values
print(forecast)
# Get the original series and the forecast
forecast_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='M')  # Adjust frequency if needed
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot original data and forecasted data
plt.figure(figsize=(10, 6))
plt.plot(df['Passengers'], label="Original Data", color='blue')
plt.plot(forecast_series, label="Forecast", color='red')
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.title("ARIMA Forecasting")
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Example: Assuming you have actual future values in 'actual_values'
# Define actual_values with the actual future values (replace with real data)
actual_values = [300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520]  # Example values

# Calculate MAE
mae = mean_absolute_error(actual_values, forecast)
mse = mean_squared_error(actual_values, forecast)
rmse = np.sqrt(mse)

print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

# Create a DataFrame for the forecast
forecast_index = pd.date_range(start='1961-01-01', periods=12, freq='MS')
forecast_series = pd.Series([446.703067, 452.850095, 456.315289, 458.939881, 
                             461.300992, 463.579517, 465.832155, 468.076679, 
                             470.318661, 472.559845, 474.800779, 477.041635], 
                            index=forecast_index)

# Plot historical and forecasted data
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Passengers'], label="Historical Data", color="blue")
plt.plot(forecast_series, label="Forecast", color="red", linestyle="dashed")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.title("ARIMA Forecast")
plt.legend()
plt.show()
