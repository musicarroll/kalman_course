#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:07:08 2023

@author: mcarroll
"""

import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

# Define the state space matrices
A = np.array([[0, 1], [-1, -1]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Create the state space model
sys = ctrl.StateSpace(A, B, C, D)

# Design a PID controller directly in state space representation
Kp = 2.0
Ki = 1.0
Kd = 0.5
A_controller = np.array([[0, 0], [0, 0]])
B_controller = np.array([[0], [0]])
C_controller = np.array([[Kd, Kp]])
D_controller = np.array([[Ki]])

controller = ctrl.StateSpace(A_controller, B_controller, C_controller, D_controller)

# Connect the controller to the system
closed_loop_sys = ctrl.feedback(controller * sys)

# Define time array for simulation
t = np.linspace(0, 10, 100)

# Simulate the system response to a step input
t, y = ctrl.step_response(closed_loop_sys, T=t)

# Plot the response
plt.plot(t, y)
plt.xlabel('Time (seconds)')
plt.ylabel('Output')
plt.title('Step Response of Controlled System')
plt.grid(True)
plt.show()
