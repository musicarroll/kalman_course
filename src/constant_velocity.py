# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:06:11 2023
constant_velocity.py:  Conversion of the matlab program
@author: cognotrend
"""


import numpy as np
import matplotlib.pyplot as plt
from collate import collate

np.random.seed(0)  # Set random seed for reproducibility

num_samples = 30
delta_t = 1

# State Transition Matrix:
Phi = np.array([[1, delta_t], [0, 1]])

# Process Noise:
Q = np.array([[0.0001, 0], [0, 0.000001]])
# Q = np.zeros((2, 2))
# Process noise sample trajectory:
w = np.random.randn(2, num_samples)
w[0] *= np.sqrt(Q[0, 0])
w[1] *= np.sqrt(Q[1, 1])

# Initial estimation uncertainty (covariance):
P0 = np.array([[0.01, 0], [0, 0.01]])

# Measurement noise covariance
R = 1**2
# Measurement matrix: Position measurement only
H = np.array([[1, 0]])

# Truth model
x_true = np.zeros((2, num_samples))
# Initial truth:
x_true[:, 0] = [0, 1] + w[:, 0]
# Apply Phi to truth and add process noise:
for k in range(1, num_samples):
    x_true[:, k] = Phi.dot(x_true[:, k - 1]) + w[:, k - 1]

# Estimate:
x_hat = np.zeros((2, num_samples))
prior_est = np.zeros((2, num_samples))  # (x-)
# Initial state:
x_hat[:, 0] = [0, 1]

# Estimation Error Covariance:
P = np.zeros((2, 2, num_samples))
prior_P = np.zeros((2, 2, num_samples))  # (P-)

# Scalar sequence to use for sawtooth plotting:
pre = np.zeros(num_samples)
post = np.zeros(num_samples)
z = np.zeros(num_samples)

# Initialize variables for the loop
P[:, :, 0] = P0
post[0] = P[0, 0, 0]
pre[0] = P[0, 0, 0]

# Initialize measurement sequence (= truth + meas noise):
z[0] = x_true[0, 0] + np.sqrt(R) * np.random.randn()

# The Kalman loop:
for k in range(1, num_samples):
    # Extrapolation:
    prior_est[:, k] = Phi.dot(x_hat[:, k - 1])  # No dynamics and noise because this is an estimate!
    prior_P[:, :, k] = Phi.dot(P[:, :, k - 1]).dot(Phi.T) + Q
    pre[k] = prior_P[0, 0, k]
    # Kalman gain:
    K = prior_P[:, :, k].dot(H.T).dot(np.linalg.inv(H.dot(prior_P[:, :, k]).dot(H.T) + R))
    # The current measurement:
    z[k] = x_true[0, k] + np.sqrt(R) * np.random.randn()
    # Update:
    x_hat[:, k] = prior_est[:, k] + K.dot(z[k] - H.dot(prior_est[:, k]))
    P[:, :, k] = (np.eye(2) - K.dot(H)).dot(prior_P[:, :, k]).dot((np.eye(2) - K.dot(H)).T) + K.dot(R).dot(K.T)
    post[k] = P[0, 0, k]

# Plot truth, measurements, and estimates.
# RMS Errors:
rms_pos = np.sqrt(np.mean((x_true[0, :] - x_hat[0, :])**2))
rms_vel = np.sqrt(np.mean((x_true[1, :] - x_hat[1, :])**2))

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(range(num_samples), x_true[0, :], 'b--*', label='Truth')
plt.plot(range(num_samples), z, 'r--o', label='Measurements')
plt.plot(range(num_samples), x_hat[0, :], 'd:k', label='Estimate')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Position (m)')
plt.title('Kalman Estimator: Non-accelerated 1D Motion')
str = ['Meas Noise Variance: {:.4f}'.format(R), 'RMS Pos Err: {:.4f}'.format(rms_pos)]
plt.annotate('\n'.join(str), xy=(0.6, 0.6), xycoords='figure fraction')
plt.subplot(2, 1, 2)
plt.plot(range(num_samples), x_true[1, :], 'b--*', label='Truth')
plt.plot(range(num_samples), x_hat[1, :], 'd:k', label='Estimate')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/sec)')
str = ['RMS Vel Err: {:.4f}'.format(rms_vel)]
plt.annotate('\n'.join(str), xy=(0.7, 0.4), xycoords='figure fraction')

print('Error:')
print(x_hat[:, num_samples - 1] - x_true[:, num_samples - 1])

# To get a sawtooth plot, you can use the imported function called collate.
# If you have two vectors of variances, v1, v2, and one is supposed to be before measurement update and one after, you create
# a new vector twice in length by v = collate(v1, v2).   Then, if t is the time vector for the period in question, you collate it with itself by
# new_time = collate(t, t).   Then you can plot v against new_time:   plot(new_time, v).

collated = collate(pre, post)
t = np.arange(1, num_samples + 1)
new_time = collate(t, t)
plt.figure(2)
plt.plot(new_time, collated)
plt.title('Variance in Position (with Process Noise)')
plt.xlabel('Time')
plt.ylabel('Variance (m^2)')
plt.show()
