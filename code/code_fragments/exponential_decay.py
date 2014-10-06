"""
This module will evaluate a exponential decay + background
signal and plot on linear, semilog, and log scales. 
Function will take the form of 
p[0] + p[1] * np.exp(-x / p[2])
""" 

from __future__ import print_function, division 
import numpy as np
import matplotlib.pyplot as plt 


def exp_decay(p,x): 
    """
    This will evaluate the exponential decay function from an array of p.
    """
    a, b, c = p

    return a + b * np.exp(-x / c)

def exp_decay_plot(p,x):
    """ 
    this will plot the exponential decay function on linear, semilogy, semilogx,
    and loglog scale. 
    """
    #run the exponential function  
    exp = exp_decay(p,x)
    
    #plot the function as a figure
    plt.figure()
    plt.suptitle(r'$y = a + b\mathrm{e}^{-x / \lambda}$', fontsize=20) 
    #define the subplots
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
   
    #plot linear scale
    ax1.plot(x, exp, 'g', alpha=0.5)
    ax1.grid(true)

     
    #plot semilogy scale
    ax2.semilogy(x, exp, 'g', alpha=0.5)
    ax2.grid(true)
    
    #plot semilogx scale
    ax3.semilogx(x, exp, 'g', alpha=0.5)
    ax3.grid(true)


    #plot loglog scale 
    ax4.loglog(x, exp, 'g', alpha=0.5)
    ax4.grid(true)


    #make it look nice. 
    plt.tight_layout(pad=0.4, h_pad=1.0)
    plt.subplots_adjust(top=0.85) #leaves space for title.  
    
    #save the figure
    plt.savefig('keg_exp_multiplot.pdf', bbox_inches='tight', dpi=300)
    

def exponential_var(p,x):
    """
    Plots the exponential decay + background function varying the values for the
    coefficient, lambda, and the noise.
    """
  
 
    plt.subplot(311)
    plt.semilogy(x, exp_decay(np.array([p[0[0]],p[1[3]],p[2[3]]])), x)
    plt.savefig('KEG_exponential_var.pdf', bbox_inches='tight', dpi=300)


