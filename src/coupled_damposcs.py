#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 12:11:35 2023
Generated with the help of chatGPT.
@author: mcarroll
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def coupled_oscillator(t, y, m1, m2, k1, k2, k3, b1, b2):
    x1, x2, v1, v2 = y
    
    dx1dt = v1
    dx2dt = v2
    
    dv1dt = (-k1 * x1 - k3 * (x1 - x2) - b1 * v1) / m1
    dv2dt = (-k2 * x2 + k3 * (x1 - x2) - b2 * v2) / m2
    
    return [dx1dt, dx2dt, dv1dt, dv2dt]

# Parameters
m1 = 1.0
m2 = 1.5
k1 = 1.2
k2 = 0.8
k3 = 0.5
b1 = 0.4
b2 = 0.6

# Initial conditions
x1_0 = 0.5
x2_0 = -0.3
v1_0 = 0.0
v2_0 = 0.0
y0 = [x1_0, x2_0, v1_0, v2_0]

# Time span
t_start = 0.0
t_end = 10.0
t_step = 0.01
t_span = np.arange(t_start, t_end, t_step)

# Solve the differential equations
sol = solve_ivp(coupled_oscillator, (t_start, t_end), y0, args=(m1, m2, k1, k2, k3, b1, b2), t_eval=t_span)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(211)
plt.plot(sol.t, sol.y[0], label='x1')
plt.plot(sol.t, sol.y[1], label='x2')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement of Masses')
plt.legend()

plt.subplot(212)
plt.plot(sol.t, sol.y[2], label='v1')
plt.plot(sol.t, sol.y[3], label='v2')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity of Masses')
plt.legend()

plt.tight_layout()
plt.show()
