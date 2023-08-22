import numpy as np
import matplotlib.pyplot as plt
from collate import collate

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
             sigma=1.0,ylim=None):
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
    plt.ylim(ylim)
    plt.show()

def tme_subplots(numsubplots, times, x_true, meas_array, x_hat, 
                 ylabels, title_string, xlabel_string):
    fig, axs = plt.subplots(numsubplots, 1, 
                figsize=(8, 8),
                sharex='col', sharey='row'
                )
    for i in range(numsubplots):
        axs[i].plot(times, x_true[i, :], 'b--*', label='Truth')
        if type(meas_array[i])!=type(None):
            axs[i].plot(times, meas_array[i], 'r--o', label='Measurements')
        axs[i].plot(times, x_hat[0, :], 'd:k', label='Estimate')
        axs[i].legend(loc='lower left')
        # axs[0].set_xlabel(f'Time (s) $\Delta_t$ ={delta_t}')
        axs[i].set_ylabel(ylabels[i])
        if i==0:
            axs[i].set_title(title_string)
            
        axs[i].set_xlabel(xlabel_string) 

def sawtooth_plots(pre,post,
                   times, title_strings, 
                   xlabel_string, 
                   ylabel_string,
                   plotnumbase=100):        
    (m,_)=pre.shape
    
    new_time = collate(times, times)
    
    for i in range(m):
        collated = collate(pre[0,:], post[0,:])
        
        plt.figure(plotnumbase+i)
        plt.plot(new_time, collated)
        plt.title(title_strings[i])
        plt.xlabel(xlabel_string)
        plt.ylabel(ylabel_string[i])
        
        plt.show()
                    
