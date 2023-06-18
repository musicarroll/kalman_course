# kf_resistor_demo.py
# Python script to demonstrate a simple, static, 1-state KF model using both measurement and process noise,
# and Kalman gain.  Full-up 1-state Kalman filter, but trivial dynamics (constant dynamics = static).

# Demo Suggestions:  Try running with 1) Initial estimates too low  2) Initial estimate too high
# 3) large sigmas  4)  small sigmas  5) large sample size  6) small sample size
import numpy as np
import matplotlib.pyplot as plt
import movavgs as ma
import math
from collate import collate

num_samples = 30
truth = 220.6  # static state model
truth_vec = truth*np.ones((num_samples,))
# State vector
x_true = truth_vec  # True state
x_hat = np.zeros((num_samples,))    # Data series to hold state estimates

# Kalman-specific parameters:
Phi = 1         # Constant dynamics
Q   = 4.0         # Process noise covariance (1x1 matrix):  Should be small since we "know" process is constant!
P0  = 1.0**2    # Initial estimation error covariance (1x1 matrix)
R   = 1.0**2    # Initial measurement noise covariance (1x1 matrix)

# Measurements: 
rg = np.random.default_rng(1)
mu=0
sigma = math.sqrt(R)
z = truth_vec + rg.normal(mu,sigma,num_samples)
x_hat[0]= z[0] # Initial state estimate / guess; could also be based on truth + init_percent*truth

H=1     # Measurement matrix (1x1) (will be no need to take transposes or inverses) 


# Process Noise stats
# initial percentage error in guess:
init_percent = 0.01
P       = np.zeros((num_samples,))  # Estimation error covariance matrix
P[0]    =  P0                       # Initial covariance
pre     = np.zeros((num_samples,))  # Prior covariance vector for plotting
post    = np.zeros((num_samples,))  # Posterior covariance for plotting
post[0] = P[0]                      # Initial pre value
pre[0]  = P[0]                      # Initial post value

# Main Kalman loop:

for k in range(1,num_samples):   # Start at k=2, because we already have k=1 as initial values
    prior_est = x_hat[k-1]      # Static problem no dynamics!
    prior_P   = P[k-1] + Q        # Temporary prior P:  Simply add process noise, since there are no dynamics
    pre[k]    = prior_P            # Set pre covariance for this k
    K         = prior_P*H/(H*prior_P*H + R)  # Compute Kalman gain; note: no inverses or transposes needed
    x_hat[k]  = prior_est + K*(z[k]-H*prior_est)  # State update using prior estimate, measurement and Kalman gain
    P[k]      = (1-K*H)*prior_P*np.transpose(1-K*H) + K*R*np.transpose(K)     # Covariance update
    post[k] = P[k]              # Set posterior vector for plotting

# Plot results:
time = np.arange(0,num_samples)


print('Estimate: ', x_hat)
print('Residual: ', x_hat-x_true)

param_string = f'Q={Q},  R={R}, x_hat[0]={round(x_hat[0],2)}, P[0]={round(P[0],2)}'
avgtype = 'Kalman Filter'
title = f'{avgtype} Estimator:  Resistance \n {param_string}' 
ylabel = 'Resistance (Ohms)'
ylim = (216,224)  # y axis scaling to be comparable with running average
ma.tme_plot(2,truth_vec, z, x_hat, ylabel, title, sigma,ylim) 


# To get a sawtooth plot, you can use the imported function called collate.
# If you have two vectors of variances, v1, v2, and one is supposed to be before measurement update and one after, you create
# a new vector twice in length by v=collate(v1,v2).   Then, if t is a the time vector for the period in question, you collate it with itself by
#new_time = collate(t,t).   Then you can plot v against new_time:   plot(new_time, v).


collated = collate(pre,post)
t=time
new_time = collate(t,t)
plt.figure(3)
plt.plot(new_time,collated)
plt.title(f'Variance in Resistance (some Process Noise)\n {param_string}')
plt.xlabel('Time')
plt.ylabel('Variance (Ohms^2)')
plt.show()
