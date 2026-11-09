#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:22:11 2023

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
num_steps = 100
time = np.arange(num_steps)
P = np.zeros(num_steps)
Buyers = np.zeros(num_steps)
Sellers = np.zeros(num_steps)

# Initial conditions
P[0] = 100.0  # Initial stock price
Buyers[0] = 60  # Initial number of buyers
Sellers[0] = 40  # Initial number of sellers

# Simulate the model
for t in range(1, num_steps):
    net_buyers = Buyers[t-1] - Sellers[t-1]

    # Update stock price
    P[t] = P[t-1] + alpha * net_buyers

    # Update number of buyers and sellers
    Buyers[t] = max(0, Buyers[t-1] + beta * P[t-1] - gamma * Sellers[t-1])
    Sellers[t] = max(0, Sellers[t-1] - beta * P[t-1] + gamma * Buyers[t-1])

# Plot stock price, number of buyers, and number of sellers
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, P, label='Stock Price', color='b')
plt.title('Stock Price vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, Buyers, label='Number of Buyers', color='g')
plt.title('Number of Buyers vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Buyers')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, Sellers, label='Number of Sellers', color='r')
plt.title('Number of Sellers vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Sellers')
plt.grid()

plt.tight_layout()
plt.show()
