#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:17:16 2023

@author: mcarroll
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
R = 1.0  # Motor resistance (ohms)
L = 0.1  # Motor inductance (H)
K_b = 0.01  # Back EMF constant (V/(rad/s))
K_f = 0.1  # Frictional constant (Nms/rad)
J = 0.01  # Moment of inertia of the load (kg*m^2)
v_app = 1.0  # Applied voltage (V)

# Initial conditions for each state variable
initial_position = 0.0  # Initial position (theta) in radians
initial_current = 0.0  # Initial current (i) in amperes
initial_angular_velocity = 0.0  # Initial angular velocity (omega) in rad/s

# Combine initial conditions into a state vector
x0 = [initial_position, initial_current, initial_angular_velocity]

# Time span
t_span = np.linspace(0, 5, 1000)  # Start at t=0 and simulate for 5 seconds

# Function to compute the state derivatives
def state_derivative(x, t):
    theta, current, omega = x
    dtheta_dt = omega
    di_dt = (v_app - R * current - K_b * omega) / L
    domega_dt = (K_b * current - K_f * omega) / J
    return [dtheta_dt, di_dt, domega_dt]

# Simulate the system using odeint
sol = odeint(state_derivative, x0, t_span)

# Extract states
position, current, angular_velocity = sol.T

# Plot the state variables
plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.plot(t_span, position, label='Position (theta)')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.legend()

plt.subplot(312)
plt.plot(t_span, current, label='Current (A)')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()

plt.subplot(313)
plt.plot(t_span, angular_velocity, label='Angular Velocity (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()

plt.tight_layout()
plt.show()
