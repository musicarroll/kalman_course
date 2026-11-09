#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:26:02 2023

@author: mcarroll
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
r_v = 0.1       # Prey intrinsic growth rate
c_vp = 0.02     # Predation rate
c_pv = 0.01     # Conversion efficiency
d_p = 0.1       # Predator death rate

# Initial populations
V0 = 40         # Initial prey population
P0 = 9          # Initial predator population

# Time settings
t_max = 200     # Maximum time
dt = 0.1        # Time step

# Lists to store population values over time
time_points = [0]
prey_population = [V0]
predator_population = [P0]

# Simulation
t = 0
V = V0
P = P0

while t < t_max:
    dVdt = r_v * V - c_vp * V * P
    dPdt = c_pv * V * P - d_p * P
    V += dVdt * dt
    P += dPdt * dt
    t += dt
    
    time_points.append(t)
    prey_population.append(V)
    predator_population.append(P)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_points, prey_population, label='Prey (Victims)')
plt.plot(time_points, predator_population, label='Predator')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.title('Predator-Prey Simulation (Lotka-Volterra Model)')
plt.legend()
plt.grid(True)
plt.show()

