#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:57:20 2023

@author: mcarroll
"""

import numpy as np
import matplotlib.pyplot as plt

# Model parameters
alpha = 0.1
beta = 0.1
gamma = 0.1
delta = 0.1
sigma = 1.0  # Price noise level

# Simulation settings
num_steps = 100
time = np.arange(num_steps)
P = np.zeros(num_steps)
Buyers = np.zeros(num_steps)
Sellers = np.zeros(num_steps)
Holders = np.zeros(num_steps)

# Initial conditions
P[0] = 100.0  # Initial stock price
Total = 100  # Initial total number of individuals
# Buyers[0] = np.random.randint(0, Total)  # Initial number of buyers
# Sellers[0] = np.random.randint(0, Total - Buyers[0])  # Initial number of sellers
# Holders[0] = Total - Buyers[0] - Sellers[0]  # Initial number of holders
Buyers[0] = Total/3.0  # Initial number of buyers
Sellers[0] = Total/3.0
Holders[0] = Total - Buyers[0] - Sellers[0]  # Initial number of holders

# Simulate the model
for t in range(1, num_steps):
    net_buyers = Buyers[t-1] - Sellers[t-1]

    # Add noise to net buyers
    net_buyers += np.random.normal(0, 5)  # Adjust the noise level as needed

    # Update stock price with noise
    P[t] = P[t-1] + alpha * net_buyers + np.random.normal(0, sigma)

    # Update number of buyers and sellers
    Buyers[t] = max(0, Buyers[t-1] + beta * P[t-1] - gamma * Sellers[t-1] - delta * Holders[t-1])
    Sellers[t] = max(0, Sellers[t-1] - beta * P[t-1] + gamma * Buyers[t-1] - delta * Holders[t-1])
    Holders[t] = Total - Buyers[t] - Sellers[t]

# Plot stock price, number of buyers, sellers, and holders
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(time, P, label='Stock Price', color='b')
plt.title('Stock Price vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(time, Buyers, label='Number of Buyers', color='g')
plt.title('Number of Buyers vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Buyers')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time, Sellers, label='Number of Sellers', color='r')
plt.title('Number of Sellers vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Sellers')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time, Holders, label='Number of Holders', color='purple')
plt.title('Number of Holders vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Holders')
plt.grid()

plt.tight_layout()
plt.show()
