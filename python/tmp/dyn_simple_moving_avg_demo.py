import numpy as np
import movavgs as ma

num_samples = 50
# Truth process (constant):

truth = 220.6
x = np.linspace(0, 4*np.pi, num_samples)
model_type = 'Dynamic'
truth_vec = truth+np.sin(x)

#model_type = 'Static'
#truth_vec = truth*np.ones(num_samples)

# Estimation sequence and initial guess:
est = np.zeros(num_samples)

# Generate noisy measurements
rg = np.random.default_rng(1)
mu=0
sigma = 0.5
meas = truth_vec + rg.normal(mu,sigma,num_samples)

# Window size:
w_size=10
gain=1/w_size

# To initialize the estimates prior to the Sw_sizeA starting,
# we just do a simple average:
est = ma.sma(meas,w_size)

if w_size<num_samples:
    avgtype = 'Simple Moving'
else:
    avgtype = 'Running'

ylabel = 'Resistance (Ohms)'

title = 'Ground Truth: ' + model_type + ' Resistance' 
ma.tme_plot(1,truth_vec, ylabel=ylabel, title=title) 
title = 'Ground Truth + Noise = Measurements: ' + model_type + ' Resistance' 
ma.tme_plot(2,truth_vec, meas, ylabel=ylabel, title=title, sigma=sigma) 
title = avgtype + ' Average Estimator: ' + model_type + ' Resistance (Window='+str(w_size)+')' 
ma.tme_plot(3,truth_vec, meas, est, ylabel, title, sigma) 

# Window size:
w_size=5
gain=1/w_size

# To initialize the estimates prior to the Sw_sizeA starting,
# we just do a simple average:
est = ma.sma(meas,w_size)

if w_size<num_samples:
    avgtype = 'Simple Moving'
else:
    avgtype = 'Running'

ylabel = 'Resistance (Ohms)'
title = avgtype + ' Average Estimator: ' + model_type + ' Resistance (Window='+str(w_size)+')' 
ma.tme_plot(4,truth_vec, meas, est, ylabel, title, sigma) 


est = ma.sma(meas,num_samples)
avgtype = 'Running'
title = avgtype + ' Average Estimator:  ' + model_type + ' Resistance ' 
ma.tme_plot(5,truth_vec, meas, est, ylabel, title, sigma) 
