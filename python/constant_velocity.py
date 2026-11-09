# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:06:11 2023
constant_velocity.py:  Conversion of the matlab program
@author: cognotrend
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collate import collate

np.random.seed(0)  # Set random seed for reproducibility

num_samples = 30
delta_t = 1

# State Transition Matrix:
Phi = np.array([[1, delta_t], [0, 1]])

# Process Noise:
Q = np.array([[0.001, 0], [0, 0.01]])
# Q = np.zeros((2, 2))
if Q[0,0]>0.0 or Q[1,1]>0:
    q_string = 'with'
else:
    q_string = 'without'

# Initial estimation uncertainty (covariance):
P0 = np.array([[4.0, 0], [0, 1.0]])

# Measurement noise covariance
R = 2**2
# Measurement matrix: Position measurement only
H = np.array([[1, 0]])

x_true = np.zeros((2, num_samples))

# True vs. assumed process noise:
# Make this parameter=1, if you want the truth and the model to have the same process
# noise stats.  Otherwise, make it any non-negative value you like for parametric studies.
process_noise_assumption_factor = 0.8 
# Process noise sample trajectory:
w1_true = process_noise_assumption_factor * np.sqrt(Q[0, 0]) * np.random.randn(num_samples) 
w2_true = process_noise_assumption_factor * np.sqrt(Q[1, 1]) * np.random.randn(num_samples)
w_true = np.vstack((w1_true, w2_true))
x_true[:, 0] = np.array([0, 1]) + w_true[:, 0] # truth at timestep 0

for k in range(1, num_samples):
    x_true[:, k] = Phi.dot(x_true[:, k-1]) + process_noise_assumption_factor*w_true[:, k-1]

# Estimate:
x_hat = np.zeros((2, num_samples))
prior_est = np.zeros((2, num_samples))  # (x-)
# Initial state:
x_hat[:, 0] = [0, 1]

# Estimation Error Covariance:
P = np.zeros((2, 2, num_samples))
prior_P = np.zeros((2, 2, num_samples))  # (P-)

# Scalar sequence to use for sawtooth plotting:
pre = np.zeros((2,num_samples))
post = np.zeros((2,num_samples))
z = np.zeros(num_samples)
matplotlib.rcParams['axes.titlesize'] = 8

# Initialize variables for the loop
P[:, :, 0] = P0
pre[0,0] = P[0, 0, 0]
pre[1,0] = P[1, 1, 0]

# Initialize measurement sequence (= truth + meas noise):
z[0] = x_true[0, 0] + np.sqrt(R) * np.random.randn()

# The Kalman loop:
for k in range(1, num_samples):
    # Extrapolation:
    prior_est[:, k] = Phi.dot(x_hat[:, k - 1])  # No dynamics and noise because this is an estimate!
    prior_P[:, :, k] = Phi.dot(P[:, :, k - 1]).dot(Phi.T) + Q
    pre[0,k] = prior_P[0, 0, k]
    pre[1,k] = prior_P[1, 1, k]
    # Kalman gain:
    K = prior_P[:, :, k].dot(H.T).dot(np.linalg.inv(H.dot(prior_P[:, :, k]).dot(H.T) + R))
    # The current measurement:
    z[k] = x_true[0, k] + np.sqrt(R) * np.random.randn()
    # Update:
    x_hat[:, k] = prior_est[:, k] + K.dot(z[k] - H.dot(prior_est[:, k]))
    P[:, :, k] = (np.eye(2) - K.dot(H)).dot(prior_P[:, :, k]).dot((np.eye(2) - K.dot(H)).T) + K.dot(R).dot(K.T)
    print(f'************** {k} *****************')
    print('prior_P:\n',prior_P[:,:,k])
    print('post_P:\n',P[:,:,k])
    post[0,k] = P[0, 0, k]
    post[1,k] = P[1, 1, k]

# Plot truth, measurements, and estimates.
# RMS Errors:
rms_pos = np.sqrt(np.mean((x_true[0, :] - x_hat[0, :])**2))
rms_vel = np.sqrt(np.mean((x_true[1, :] - x_hat[1, :])**2))

# Customm TME Plot:  Hasn't yet been integrated with movavgs.tme_plot()

matplotlib.rcParams['axes.titlesize'] = 8


posQ = f'Pos Q={np.round(Q[0,0],4)}'
velQ = f'Vel Q={np.round(Q[1,1],8)}'
pos_init_est = f'Pos={np.round(x_hat[0,0],2)}'
vel_init_est = f'Vel={np.round(x_hat[1,0],2)}'
param_string = f'Model Init: {pos_init_est}, {vel_init_est}, {posQ}, {velQ},\n Pos P={np.round(P[0,0,0],4)}, Vel P={np.round(P[1,1,0],4)}, R={R}'

# param_string = f'Pos Q[0]={Q[0,0]}, Vel Q[0]={Q[1,1]}  R={R},\n Pos x_hat[0]={np.round(x_hat[0,0],2)}, Vel x_hat[0]={np.round(x_hat[0,1],2)}, \n Pos P[0]={np.round(P[0,0,0],2)}, Vel P[0]={np.round(P[1,1,0],2)}'
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(range(num_samples), x_true[0, :], 'b--*', label='Truth')
plt.plot(range(num_samples), z, 'r--o', label='Measurements')
plt.plot(range(num_samples), x_hat[0, :], 'd:k', label='Estimate')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Position (m)')
plt.title('Kalman Estimator: Non-accelerated 1D Motion\n'+ param_string)
str = ['RMS Pos Err: {:.4f}'.format(rms_pos)]
plt.annotate('\n'.join(str), xy=(0.2, 0.55), xycoords='figure fraction')
plt.subplot(2, 1, 2)
plt.plot(range(num_samples), x_true[1, :], 'b--*', label='Truth')
plt.plot(range(num_samples), x_hat[1, :], 'd:k', label='Estimate')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/sec)')
str = ['RMS Vel Err: {:.4f}'.format(rms_vel)]
plt.annotate('\n'.join(str), xy=(0.65, 0.42), xycoords='figure fraction')

print('Error:')
print(x_hat[:, num_samples - 1] - x_true[:, num_samples - 1])

# To get a sawtooth plot, you can use the imported function called collate.
# If you have two vectors of variances, v1, v2, and one is supposed to be before measurement update and one after, you create
# a new vector twice in length by v = collate(v1, v2).   Then, if t is the time vector for the period in question, you collate it with itself by
# new_time = collate(t, t).   Then you can plot v against new_time:   plot(new_time, v).

collated = collate(pre[0,:], post[0,:])
t = np.arange(1, num_samples + 1)
new_time = collate(t, t)
plt.figure(2)
plt.plot(new_time, collated)
plt.title(f'Variance in Position ({q_string} Process Noise)\n'+param_string)
plt.xlabel('Time')
plt.ylabel('Variance (m^2)')
plt.show()


# Sawtooth of velocity variance:

collated = collate(pre[0,:], post[1,:])
t = np.arange(1, num_samples + 1)
new_time = collate(t, t)
plt.figure(3)
plt.plot(new_time, collated)
plt.title(f'Variance in Velocity ({q_string} Process Noise)\n'+param_string)
plt.xlabel('Time')
plt.ylabel('Variance ((m/s)^2)')
plt.show()
    