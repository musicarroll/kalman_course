#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:59:36 2023

@author: mcarroll
"""

import numpy as np
import matplotlib.pyplot as plt

# Model parameters
alpha = 1.1
beta = 0.01
gamma = 0.01
delta = 0.9


# Simulation settings
num_steps = 20
time = np.arange(num_steps)
P = np.zeros(num_steps)
V = np.zeros(num_steps)

# Initial conditions
P[0] = 100.0  # Initial stock price
V[0] = 50000.0   # Initial trading volume

# Simulate the model
for t in range(1, num_steps):
    P[t] = alpha * P[t-1] - beta * P[t-1] * V[t-1]
    V[t] = gamma * P[t-1] * V[t-1] - delta * V[t-1]

# Plot stock price and trading volume
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, P, label='Stock Price', color='b')
plt.title('Stock Price vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, V, label='Trading Volume', color='g')
plt.title('Trading Volume vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Trading Volume')
plt.grid()

plt.tight_layout()
plt.show()
