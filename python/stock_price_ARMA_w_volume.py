#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 06:22:32 2023

@author: mcarroll
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulate an ARMA model with volume influence
np.random.seed(0)
n = 100
phi1 = 0.5
phi2 = -0.2
theta1 = 0.7
theta2 = -0.3
eta = 0.2

Y = np.zeros(n)
V = np.random.normal(0, 1, n)
eps = np.random.normal(0, 1, n)

for t in range(2, n):
    Y[t] = phi1 * Y[t-1] + phi2 * Y[t-2] + theta1 * eps[t-1] + theta2 * eps[t-2] + eta * V[t]

# Plot stock prices and trading volume
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(Y, label='Stock Price')
plt.title('Simulated Stock Prices with Volume Influence')
plt.ylabel('Price')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(V, label='Trading Volume', color='red')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.grid()

plt.show()
