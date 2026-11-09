#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:59:57 2023

@author: mcarroll
"""

import numpy as np
import statsmodels.api as sm

# Generate some random data
np.random.seed(0)
n = 100
time = np.arange(n)
observed_data = 0.5 * time + np.random.normal(scale=0.2, size=n)

# Define the state space model
mod = sm.tsa.statespace.SARIMAX(
    observed_data, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)
)

# Estimate the model parameters
res = mod.fit()

# Get the predicted values
predicted_values = res.get_prediction(start=0, end=n).predicted_mean

# Print the model summary
print(res.summary())
