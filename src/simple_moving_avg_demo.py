import numpy as np
import movavgs as ma


mysigma = 0.5
num_samples = 50
# Truth process (constant):
truth = 220.6
truth_vec = truth*np.ones(num_samples)
# Estimation sequence and initial guess:
# Estimation sequence and initial guess:
est = np.zeros(num_samples)

# Generate noisy measurements
rg = np.random.default_rng(1)
mu=0
sigma = 0.5
meas = truth_vec + rg.normal(mu,sigma,num_samples)

# Window size:
w_size=20
gain=1/w_size

# To initialize the estimates prior to the Sw_sizeA starting,
# we just do a simple average:
est = ma.sma(meas,w_size)

if w_size<num_samples:
    avgtype = 'Simple Moving'
else:
    avgtype = 'Running'

ylabel = 'Resistance (Ohms)'
title = avgtype + ' Average Estimator: Resistance (Window='+str(w_size)+')' 
ma.tme_plot(1,truth_vec, meas, est, ylabel, title, sigma) 
