#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:03:09 2023

@author: mcarroll
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.statespace.kalman_filter as kf

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic stock price data
num_days = 500
mean_return = 0.0005  # Average daily return (drift)
daily_volatility = 0.02  # Daily volatility
closing_prices = np.cumprod(1 + np.random.normal(mean_return, daily_volatility, num_days))
closing_prices = 100 * closing_prices

# Create a date index for the time series
date_index = pd.date_range(start='2022-01-01', periods=num_days, freq='D')

# Create a DataFrame to store the stock price data
stock_data = pd.DataFrame({'Date': date_index, 'Closing Price': closing_prices})

# Implement the Stochastic Volatility (SV) model with Kalman Filter
log_returns = np.log(stock_data['Closing Price'] / stock_data['Closing Price'].shift(1)).dropna()
observed_returns = np.sqrt(np.exp(log_returns))

# Define the state space matrices for the Kalman Filter
def state_transition(epsilon, phi, log_sigma):
    return np.log(epsilon) + phi * (log_sigma - np.log(epsilon))
sm.tsa.statespace.kalman_filter
def observation(volatility):
    return volatility

# Initialize the Kalman Filter parameters
epsilon_initial = daily_volatility  # Initial guess for epsilon
phi_initial = 0.98  # Initial guess for phi
sigma_initial = np.std(log_returns)  # Initial guess for log-sigma (volatility)

# Define the Kalman Filter model
kalman_model = kf.KalmanFilter(
    observed_returns,
    k_states=1,
    k_posdef=1,
    initialization='approximate_diffuse',
    initial_state_covariance=1000 * np.eye(1),
    state_transition=state_transition,
    observation=observation,
    initial_state= np.array([np.log(epsilon_initial ** 2)]),
    initialization_warnings=False,
    initial_state_covariance_fixed=True
)

# Fit the Kalman Filter model
kalman_results = kalman_model.fit()

# Get the estimated volatility (hidden state)
estimated_volatility = np.sqrt(np.exp(kalman_results.filter_results.filtered_state))

# Plot the original stock returns and the estimated volatility
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'][1:], observed_returns, label='Observed Returns', color='blue')
plt.plot(stock_data['Date'][1:], estimated_volatility, label='Estimated Volatility', color='red')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Observed Returns and Estimated Volatility using Kalman Filter')
plt.legend()
plt.grid(True)
plt.show()
