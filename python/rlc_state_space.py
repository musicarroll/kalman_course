#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 09:03:42 2023

@author: mcarroll
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the state space model parameters
R = 1.0  # Resistance
L = 1.0  # Inductance
C = 1.0  # Capacitance

# Define the state space matrices
A = np.array([[-1/(R*C), -1/C], [1/L, -R/L]])
B = np.array([[1/(R*C)], [0]])

# # Input function (step input for simplicity)
# def input_function(t):
#     return 1.0 if 1.0 <= t <= 2.0 else 0.0  # Single pulse from t=1 to t=2
#     # return 1.0 if t >= 0 else 0.0

# Define the square wave input function
def input_function(t):
    period = 2.0  # Period of the square wave
    duty_cycle = 0.75  # Duty cycle of the square wave
    t_in_period = t % period
    return 1.0 if t_in_period < period * duty_cycle else 0.0


# Define the system dynamics
def system_dynamics(state, t):
    Vc, IL = state
    Vin = input_function(t)
    dVc_IL_dt = np.dot(A, np.array([[Vc], [IL]])) + np.dot(B, Vin)
    # print(dVc_IL_dt.shape)
    # print(dVc_IL_dt.flatten().shape)
    return dVc_IL_dt.flatten()  # Return a one-dimensional array

# Initial conditions for both variables
initial_state = np.array([0.0, 0.0])

# Time points for simulation
time_points = np.linspace(0, 10, 1000)

# Solve the ODE system
solution = odeint(system_dynamics, initial_state, time_points)

# Extract the simulated voltages and currents
Vc_simulated = solution[:, 0]
IL_simulated = solution[:, 1]

# Plot the results
plt.plot(time_points, Vc_simulated, label='Capacitor Voltage (Vc)')
plt.plot(time_points, IL_simulated, label='Inductor Current (IL)')
plt.xlabel('Time')
plt.ylabel('Voltage/Current')
plt.legend()
plt.show()
