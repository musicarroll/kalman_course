# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:24:20 2023
Conversion of damposc.m to python.
@author: cognotrend
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from collate import collate

num_samples = 100
delta_t = .25
m = 1
damp_coeff   = 0.1
spring_coeff = 0.1

Phi = expm(np.array([[0, 1], [-spring_coeff/m, -damp_coeff/m]]) * delta_t)

Q = np.diag([0.1**2, 0.01**2])
P0 = np.diag([0.01**2, 0.01**2])
R = 0.5**2
H = np.array([1, 0])
w1 = np.random.randn(num_samples) * np.sqrt(Q[0, 0])
w2 = np.random.randn(num_samples) * np.sqrt(Q[1, 1])
w = np.vstack((w1, w2))
x_true = np.zeros((2, num_samples))
process_noise_assumption_factor = 0.5  # factor by which process noise assumption is off
x_true[:, 0] = np.array([1, 0]) + process_noise_assumption_factor*w[:, 0]

for k in range(1, num_samples):
    x_true[:, k] = Phi.dot(x_true[:, k-1]) + process_noise_assumption_factor*w[:, k-1]

x_hat = np.zeros((2, num_samples))
prior_est = np.zeros((2, num_samples))
x_hat[:, 0] = np.array([1, 0])

P = np.zeros((2, 2, num_samples))
prior_P = np.zeros((2, 2, num_samples))
pre = np.zeros(num_samples)
post = np.zeros(num_samples)
z = np.zeros(num_samples)

P[:, :, 0] = P0
post[0] = P[0, 0, 0]
pre[0] = P[0, 0, 0]
z[0] = x_true[0, 0] + np.sqrt(R) * np.random.randn()

for k in range(1, num_samples):
    prior_est[:, k] = Phi.dot(x_hat[:, k-1])
    prior_P[:, :, k] = Phi.dot(P[:, :, k-1]).dot(Phi.T) + Q
    pre[k] = prior_P[0, 0, k]
    K = prior_P[:, :, k].dot(H.T) / (H.dot(prior_P[:, :, k]).dot(H.T) + R)
    z[k] = x_true[0, k] + np.sqrt(R) * np.random.randn()
    x_hat[:, k] = prior_est[:, k] + K * (z[k] - H.dot(prior_est[:, k]))
    P[:, :, k] = (np.eye(2) - K.dot(H)).dot(prior_P[:, :, k]).dot((np.eye(2) - K.dot(H)).T) + K.dot(R).dot(K.T)
    post[k] = P[0, 0, k]

rms_pos = np.sqrt(np.mean((x_true[0, :] - x_hat[0, :])**2))
rms_vel = np.sqrt(np.mean((x_true[1, :] - x_hat[1, :])**2))

param_string = f'Pos Q[0]={np.round(Q[0,0],4)}, Vel Q[0]={np.round(Q[1,1],4)}  R={R},\n Pos x_hat[0]={np.round(x_hat[0,0],2)}, Vel x_hat[0]={np.round(x_hat[0,1],2)}, \n Pos P[0]={np.round(P[0,0,0],4)}, Vel P[0]={np.round(P[1,1,0],4)}'
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
times = delta_t*np.arange(1, num_samples+1)
axs[0].plot(times, x_true[0, :], 'b--*', label='Truth')
axs[0].plot(times, z, 'r--o', label='Measurements')
axs[0].plot(times, x_hat[0, :], 'd:k', label='Estimate')
axs[0].legend(loc='lower left')
axs[0].set_xlabel(f'Time (s) delta_t ={delta_t}')
axs[0].set_ylabel('Position (m)')
axs[0].set_title('Kalman Estimator: Damped Harmonic Oscillator\n'+param_string)
str_text = f"Meas Noise Var: {np.round(R,4)}\nRMS Pos Err: {np.round(rms_pos,4)}"
axs[0].text(0.7, 0.8, str_text, transform=axs[0].transAxes)
axs[1].plot(times, x_true[1, :], 'b--*', label='Truth')
axs[1].plot(times, x_hat[1, :], 'd:k', label='Estimate')
axs[1].legend(loc='lower left')
axs[1].set_xlabel(f'Time (s) delta_t ={delta_t}') 
axs[1].set_ylabel('Velocity (m/sec)')
str_text = f"RMS Vel Err: {np.round(rms_vel,4)}"
axs[1].text(0.7, 0.4, str_text, transform=axs[1].transAxes)

print('Error:')
print(x_hat[:, -1] - x_true[:, -1])

collated = collate(pre, post)

new_time = collate(times, times)
plt.figure()
plt.plot(new_time, collated)
plt.title('Variance in Position Estimate (with Process Noise)\n'+param_string)
plt.xlabel(f'Time(s) delta_t ={delta_t} ')
plt.ylabel('Variance (m^2)')

plt.show()

