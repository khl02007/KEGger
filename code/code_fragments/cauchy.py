"""
This module plots the cauchy distribution. The plotted function is

p[0] / (np.pi * (p[0]**2 + (x - p[1])**2)
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def cauchy_dist(p, x):
    """ 
    Returns p[0] /  (np.pi * (p[0]**2 + (x - p[1])**2)
    """
    
    b, a = p
    return b / (np.pi * (b**2 + (x - a)**2))
    
def cauchy_mulitplot(p, x):
    """
    Plots the Cauchy distribution on linear, semilogx, semilogy, and
    loglog scales.
    """
    
    cauchy_run = cauchy_dist(p, x)
   
    #Define the figure
    plt.figure()
    plt.suptitle(r'$y = \frac{\beta}{\pi(\beta^2 + (x-\alpha)^2)}$', \
            fontsize=22)
    #Plots on the linear scale
    plt.subplot(221)
    plt.plot(x, cauchy_run, 'k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid(True)
    
    #Plots on the semilogy scale
    plt.subplot(222)
    plt.semilogy(x, cauchy_run, 'k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid(True)
    
    #Plots on the semilogx scale
    plt.subplot(223)
    plt.semilogx(x, cauchy_run, 'k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid(True)
    
    #Plots on the loglog scale
    plt.subplot(224)
    plt.loglog(x, cauchy_run, 'k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid(True)
    
    #Make it look nice
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    #Save the figure
    plt.savefig('KEG_cauchy_multiplot.pdf', bbox_inches='tight', dpi=300)
