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

num_samples = 30
delta_t = 0.25
m = 1
damp_coeff   = 0.1
spring_coeff = 0.1

Phi = expm(np.array([[0, 1], [-spring_coeff/m, -damp_coeff/m]]) * delta_t)

Q = np.diag([0.0001**2, 0.0001**2])
P0 = np.diag([0.0001**2, 0.0001**2])
R = 0.00000001**2
H = np.array([1, 0])
w1 = np.random.randn(num_samples) * np.sqrt(Q[0, 0])
w2 = np.random.randn(num_samples) * np.sqrt(Q[1, 1])
w = np.vstack((w1, w2))
x_true = np.zeros((2, num_samples))
x_true[:, 0] = np.array([1, 0]) + w[:, 0]

for k in range(1, num_samples):
    x_true[:, k] = Phi.dot(x_true[:, k-1]) + w[:, k-1]

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

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(range(1, num_samples+1), x_true[0, :], 'b--*', label='Truth')
axs[0].plot(range(1, num_samples+1), z, 'r--o', label='Measurements')
axs[0].plot(range(1, num_samples+1), x_hat[0, :], 'd:k', label='Estimate')
axs[0].legend(loc='lower right')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position (m)')
axs[0].set_title('Kalman Estimator: Damped Harmonic Oscillator')
str_text = f"Meas Noise Variance: {R}\nRMS Pos Err: {rms_pos}"
axs[0].text(0.7, 0.8, str_text, transform=axs[0].transAxes)
axs[1].plot(range(1, num_samples+1), x_true[1, :], 'b--*', label='Truth')
axs[1].plot(range(1, num_samples+1), x_hat[1, :], 'd:k', label='Estimate')
axs[1].legend(loc='lower right')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Velocity (m/sec)')
str_text = f"RMS Vel Err: {rms_vel}"
axs[1].text(0.7, 0.4, str_text, transform=axs[1].transAxes)

print('Error:')
print(x_hat[:, -1] - x_true[:, -1])

collated = collate(pre, post)
t = np.arange(1, num_samples+1)
new_time = collate(t, t)
plt.figure()
plt.plot(new_time, collated)
plt.title('Variance in Position (with Process Noise)')
plt.xlabel(f'Time Steps with delta_t ={delta_t} ')
plt.ylabel('Variance (m^2)')

plt.show()

