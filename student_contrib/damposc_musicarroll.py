# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:24:20 2023
Conversion of damposc.m to python.
@author: cognotrend
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import expm
from collate import collate

np.random.seed(0)  # Set random seed for reproducibility

num_samples =50
delta_t = 1
m = 1 # kg
spring_coeff = 1 # N/m light spring
damp_coeff   = 5 # Ns/m medium damping

Phi = expm(np.array([[0, 1], 
        [-spring_coeff/m, -damp_coeff/m]]) * delta_t)

# Model statistics:
q_cont = 0.01 # contuous process noise variance 
Q = q_cont*np.array([[delta_t**3/3.0, delta_t**2/2.0],
                     [delta_t**2/2.0,delta_t]])
P0 = np.diag([0.01**2, 0.1**2])
# Measurement noise covariance
R = np.array([[0.01,0],[0,0.001]])
# Measurement matrix: Both Position and velocity measurements only
H = np.eye(2)
x_true = np.zeros((2, num_samples))

# True vs. assumed process noise:
# Make this parameter=1, if you want the truth and the model to have the same process
# noise stats.  Otherwise, make it any non-negative value you like for parametric studies.
process_noise_assumption_factor = 0.8  
w1_true = process_noise_assumption_factor * np.sqrt(Q[0, 0]) * np.random.randn(num_samples) 
w2_true = process_noise_assumption_factor * np.sqrt(Q[1, 1]) * np.random.randn(num_samples)
w_true = np.vstack((w1_true, w2_true))
x_true[:, 0] = np.array([1, 0]) + w_true[:, 0] # truth at timestep 0

for k in range(1, num_samples):
    x_true[:, k] = Phi.dot(x_true[:, k-1]) + process_noise_assumption_factor*w_true[:, k-1]

x_hat = np.zeros((2, num_samples))
prior_est = np.zeros((2, num_samples))
x_hat[:, 0] = np.array([1, 0])

P = np.zeros((2, 2, num_samples))
prior_P = np.zeros((2, 2, num_samples))
pre = np.zeros((2,num_samples))
post = np.zeros((2,num_samples))
z = np.zeros((2,num_samples))

# Initialize measurement sequence (= truth + meas noise):
z[0,0] = x_true[0, 0] + np.sqrt(R[0,0]) * np.random.randn()
z[1,0] = x_true[1, 0] + np.sqrt(R[1,1]) * np.random.randn()

P[:, :, 0] = P0
post[0,0] = P[0, 0, 0]
post[1,0] = P[1,1,0]
pre[0,0] = P[0, 0, 0]
pre[1,0] = P[1, 1, 0]


for k in range(1, num_samples):
    prior_est[:, k] = Phi.dot(x_hat[:, k-1])
    prior_P[:, :, k] = Phi.dot(P[:, :, k-1]).dot(Phi.T) + Q
    pre[0,k] = prior_P[0, 0, k]
    pre[1,k] = prior_P[1, 1, k]
    K = prior_P[:, :, k].dot(H.T).dot(np.linalg.inv(H.dot(prior_P[:, :, k]).dot(H.T) + R))
    # K = prior_P[:, :, k].dot(H.T) / (H.dot(prior_P[:, :, k]).dot(H.T) + R)
    z[0,k] = x_true[0, k] + np.sqrt(R[0,0]) * np.random.randn()
    z[1,k] = x_true[1, k] + np.sqrt(R[1,1]) * np.random.randn()
    x_hat[:, k] = prior_est[:, k] + K.dot(z[:,k] - H.dot(prior_est[:, k]))
    P[:, :, k] = (np.eye(2) - K.dot(H)).dot(prior_P[:, :, k]).dot((np.eye(2) - K.dot(H)).T) + K.dot(R).dot(K.T)
    print(f'************** {k} *****************')
    print('prior_P:\n',prior_P[:,:,k])
    print('post_P:\n',P[:,:,k])
    post[0,k] = P[0, 0, k]
    post[1,k] = P[1, 1, k]


rms_pos = np.sqrt(np.mean((x_true[0, :] - x_hat[0, :])**2))
rms_vel = np.sqrt(np.mean((x_true[1, :] - x_hat[1, :])**2))
matplotlib.rcParams['axes.titlesize'] = 8


posQ = f'Pos Q={np.round(Q[0,0],4)}'
velQ = f'Vel Q={np.round(Q[1,1],8)}'
pos_init_est = f'Pos={np.round(x_hat[0,0],2)}'
vel_init_est = f'Vel={np.round(x_hat[1,0],2)}'
PosR = np.round(R[0,0],4)
VelR = np.round(R[1,1],4)
param_string = f'Model Init: {pos_init_est}, {vel_init_est}, {posQ}, {velQ},\n Pos P={np.round(P[0,0,0],4)}, Vel P={np.round(P[1,1,0],4)}, Pos R={PosR}, Vel R={VelR}'


fig, axs = plt.subplots(2, 1, figsize=(8, 8))
times = delta_t*np.arange(1, num_samples+1)
axs[0].plot(times, x_true[0, :], 'b--*', label='Truth')
axs[0].plot(times, z[0,:], 'r--o', label='Measurements')
axs[0].plot(times, x_hat[0, :], 'd:k', label='Estimate')
axs[0].legend(loc='lower left')
axs[0].set_xlabel(f'Time (s) $\Delta_t$ ={delta_t}')
axs[0].set_ylabel('Position (m)')
axs[0].set_title('Kalman Estimator: Damped Harmonic Oscillator w/ Pos and Vel Sensors\n'+param_string)
str_text = f"RMS Pos Err: {np.round(rms_pos,4)}"
axs[0].text(0.5, 0.1, str_text, transform=axs[0].transAxes)
axs[1].plot(times, x_true[1, :], 'b--*', label='Truth')
axs[1].plot(times, z[1,:], 'r--o', label='Measurements')
axs[1].plot(times, x_hat[1, :], 'd:k', label='Estimate')
axs[1].legend(loc='lower left')
axs[1].set_xlabel(f'Time (s), $\Delta t$={delta_t}') 
axs[1].set_ylabel('Velocity (m/sec)')
str_text = f"RMS Vel Err: {np.round(rms_vel,4)}"
axs[1].text(0.5, 0.1, str_text, transform=axs[1].transAxes)

print('Error:')
print(x_hat[:, -1] - x_true[:, -1])

collated = collate(pre[0,:], post[0,:])
t = np.arange(1, num_samples + 1)
new_time = collate(t, t)
plt.figure(2)
plt.plot(new_time, collated)
plt.title(f'Variance in Position with Pos and Vel Sensors\n{param_string}')
plt.xlabel(f'Time(s), $\Delta t$={delta_t} ')
plt.ylabel('Variance (m^2)')
plt.show()

# Sawtooth of velocity variance:

collated = collate(pre[0,:], post[1,:])
t = np.arange(1, num_samples + 1)
new_time = collate(t, t)
plt.figure(3)
plt.plot(new_time, collated)
plt.title(f'Variance in Velocity and Vel Sensor\n{param_string}')
plt.xlabel(f'Time(s), $\Delta t$={delta_t} ')
plt.ylabel('Variance ((m/s)^2)')
plt.show()
    