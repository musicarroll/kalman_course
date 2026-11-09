#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:23:01 2023

@author: mcarroll
"""

import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

# Define the state space matrices of the system (plant)
A = np.array([[0, 1], [-1, -1]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Create the state space model of the system
sys = ctrl.StateSpace(A, B, C, D)


# Define the covariance matrices for process noise and measurement noise
QN = np.array([[0.01, 0], [0, 0.01]])  # Full process noise covariance matrix
RN = np.array([[0.1]])  # Measurement noise covariance
G=np.eye(2)
# Design the Kalman filter (linear quadratic estimator)
#kalman_gain, P, E = ctrl.lqe(sys, Q, R)
L, P, E = ctrl.lqe(A, G, C, QN, RN)
# The Kalman gain matrix is available for state estimation
print("Steady-state Kalman Gain Matrix:")
print(L)

