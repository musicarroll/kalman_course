import numpy as np
import movavgs as ma
import math

num_samples = 30

# Truth process (constant):
truth = 220.6
rg = np.random.default_rng(1)
mu=0
Q = 0.1**2
sigma = math.sqrt(Q)
# Add process noise to truth model:
truth_vec = truth*np.ones(num_samples) + rg.normal(mu,sigma,num_samples)
# Estimation sequence and initial guess:
est = np.zeros(num_samples)

# Window size:
# Running average:  SMA window size=num_samples
w_size=num_samples
gain=1/w_size

# Generate noisy measurments
rg = np.random.default_rng(1)
mu=0
sigma = 1.0
meas = truth_vec + rg.normal(mu,sigma,num_samples)
est = ma.sma(meas,w_size)

param_string = f'R={sigma**2}'
if w_size<num_samples:
    avgtype = 'Simple Moving'
    title = avgtype + ' Average Estimator: Static Resistance (Window='+str(w_size)+')' 
else:
    avgtype = 'Running'
    title = f'{avgtype} Average Estimator: Static Resistance\n {param_string}' 
        

ylabel = 'Resistance (Ohms)'
ylim = (216,224)
ma.tme_plot(1,truth_vec, meas, est, ylabel, title, sigma, ylim)
