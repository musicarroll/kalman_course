#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 09:42:09 2023

@author: mcarroll
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the state space model parameters for the simplified model
R1 = 1.0  # Resistance 1
R2 = 2.0  # Resistance 2
L1 = 1.0  # Inductance 1
L2 = 0.5  # Inductance 2
C1 = 1.0  # Capacitance 1
C2 = 0.5  # Capacitance 2

# Define the state space matrices
A = np.array([
    [-1/(R1*C1), 0, -1/C1, 0],
    [0, -R2/L2, 0, 0],
    [1/L1, 0, -1/(R1*C1), -1/C1],
    [0, 0, 1/L2, -R2/L2]
])

# Define the simplified B matrix with only a voltage input
B = np.array([
    [1/(R1*C1)],
    [0],
    [0],
    [0]
])

# Input function (square wave input)
def input_function(t):
    period = 2.0  # Period of the square wave
    duty_cycle = 0.5  # Duty cycle of the square wave
    t_in_period = t % period
    return np.array([[1.0 if t_in_period < period * duty_cycle else 0.0]])

# Define the system dynamics
def system_dynamics(state, t):
    Vin = input_function(t)
    dstate_dt = np.dot(A, state) + np.dot(B, Vin)
    return dstate_dt

# Initial conditions for all variables
initial_state = np.array([0.0, 0.0, 0.0, 0.0])

# Time points for simulation (increased to 20 seconds)
time_points = np.linspace(0, 20, 2000)

# Solve the ODE system
solution = odeint(system_dynamics, initial_state, time_points)

# Extract the simulated variables
Vc1_simulated = solution[:, 0]
IL1_simulated = solution[:, 1]
Vc2_simulated = solution[:, 2]
IL2_simulated = solution[:, 3]

# Plot the results
plt.plot(time_points, Vc1_simulated, label='Capacitor 1 Voltage (Vc1)')
plt.plot(time_points, IL1_simulated, label='Inductor 1 Current (IL1)')
plt.plot(time_points, Vc2_simulated, label='Capacitor 2 Voltage (Vc2)')
plt.plot(time_points, IL2_simulated, label='Inductor 2 Current (IL2)')
plt.xlabel('Time')
plt.ylabel('Voltage/Current')
plt.legend()
plt.show()
