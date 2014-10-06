"""
This module plots the exponential decay function + background noise
on a semilogy plot. The plotted function is

p[0] + p[1] * np.exp(-x / p[2])
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt 

def exp_decay(p, x):
    """
    Returns the exponential decay function with background.
    p[0] + p[1] * np.exp(-x / p[3])
    """
    a, b, c = p

    return a + b * np.exp(-x / c)

def plot_exp_decay_semilogy(p, x):
    """
    This function will plot the exponential decay function
    on a semilogy plot. 
    """
    #Run the exponential decay given the above parameters.
    exp_decay_run = exp_decay(p, x)

    #Generate the plot on semilogy
    plt.figure()
    plt.semilogy(x, exp_decay_run, 'g.', alpha=0.5)
    plt.grid(True)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$y = a + b\times\mathrm{e}^{-x / \lambda}$')
  

    #This will save the figure as an eps document in the working
    #directory
    plt.savefig('exp_decay_plots_tightlayout2.pdf', bbox_inches='tight', transparent=True, dpi=300)
