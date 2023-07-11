#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:20:50 2023

@author: mcarroll
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE function
def ode_func(t, y):
    return -2 * y  # Example: ODE dy/dt = -2y

# Define the initial conditions
y0 = 1.0  # Initial value of y at t=0
t_span = (0, 5)  # Time span to solve the ODE over

# Solve the ODE
sol = solve_ivp(ode_func, t_span, [y0])

# Extract the solution
t = sol.t
y = sol.y[0]

# Plot the solution
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Solution of ODE')
plt.show()
