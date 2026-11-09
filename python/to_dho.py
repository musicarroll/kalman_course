#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:29:09 2024

@author: mcarroll
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# System parameters
m = 1.0  # Mass
k = 10.0  # Spring constant
b = 0.5  # Damping coefficient

# State-space matrices
A = np.array([[0, 1, 0],
              [-k/m, -b/m, 0],
              [b*k/m**2,b**2/m**2-k/m,0]])
B = np.array([[0], [0], [1]])
C = np.array([1, 0, 0])
D = 0

# Initial conditions
x0 = np.array([1.0, 0.0, 0.0])

# Time vector
t = np.linspace(0, 10, 1000)

# Define the system dynamics
def system(x, t):
    return A.dot(x)

# Simulate the system
x = odeint(system, x0, t)

# Extract position for plotting
y = C.dot(x.T).flatten()

# Plot the results
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Position (x)')
plt.title('Third Order State-Space Model Simulation')
plt.grid(True)
plt.show()
