import numpy as np
import matplotlib.pyplot as plt

def sma(data,w_size):
    n=data.size # Assumes data is numpy array
    smaval = np.zeros(n)
    gain = 1/w_size
    for i in range(n):
        if i<w_size:
            smaval[i] = data[0:i+1].mean()
        else:
            smaval[i] = smaval[i-1] + gain*(data[i] - smaval[i-w_size+1])
    return smaval

def recursive_moving_avg(meas,gain,prior_est,oldest_meas):
    avg = prior_est + gain*(meas-oldest_meas)
    return avg

def tme_plot(plotnum,truth_data, meas_data=None, 
             est_data=None, ylabel='y', title='', 
             sigma=1.0):
    plt.figure(plotnum)
    plt.plot(truth_data,'.-b')
    legend = []
    legend.append('Truth')
    if type(meas_data)!=type(None):
        plt.plot(meas_data,'.r')
        legend.append('Measurements ($\sigma$='+str(sigma)+')')
    if type(est_data)!=type(None):
        plt.plot(est_data,'+:k')
        legend.append('Estimates')
    plt.xlabel('Time (Epochs)')
    plt.ylabel(ylabel)
    plt.legend(legend,loc='lower left')
    plt.title(title)
    plt.show()


