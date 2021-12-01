import numpy as np
import movavgs as ma

num_samples = 50

# Truth process (constant):
truth = 220.6
truth_vec = truth*np.ones(num_samples)
# Estimation sequence and initial guess:
est = np.zeros(num_samples)

# Window size:
# Running average:  SMA window size=num_samples
w_size=num_samples
gain=1/w_size

# Generate noisy measurments
rg = np.random.default_rng(1)
mu=0
sigma = 0.5
meas = truth_vec + rg.normal(mu,sigma,num_samples)
est = ma.sma(meas,w_size)

if w_size<num_samples:
    avgtype = 'Simple Moving'
    title = avgtype + ' Average Estimator: Static Resistance (Window='+str(w_size)+')' 
else:
    avgtype = 'Running'
    title = avgtype + ' Average Estimator: Static Resistance' 
        

ylabel = 'Resistance (Ohms)'
ma.tme_plot(1,truth_vec, meas, est, ylabel, title, sigma)
