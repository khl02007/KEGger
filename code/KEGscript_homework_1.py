"""
This script generates all figures and analysis for homework 1 of bibe103 at
Caltech. This is is collective work of Emily Blythe (EB), Griffin Chure (GC) and
Kyu Hyun Lee (KL). While all modules were written collectively, modules written
primarily by one person are indicated with the authors' initials. 
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
 

# ######################
# Code for problem 1.2 #
# ###################### 

"""
The following modules define the exponential decay + background function, 
Cauchy distribution, and the Hill function to be used for parts a, b, and c
respectively.
""" 

# ################
def exp_decay(p, x): 
    """
    (GC)
    This will evaluate the exponential decay function from an array of p. 
    The form of the function is 
    p[0] + p[1] * np.exp(-x / p[2])
    """
    a, b, lam = p

    if lam == 0: 
        raise ValueError("You can't divide by zero, stupid.")
    else:
        return a + b * np.exp(-x / lam)

# ###############
def cauchy(p, x):
    """ 
    (EB)
    Evaluates the Cauchy distribution from an array of p. The form of the\
    function is 
    p[0] /  (np.pi * (p[0]**2 + (x - p[1])**2)
    """
    
    b, a = p
    return b / (np.pi * (b**2 + (x - a)**2))
 

# ###############
def hill(p, x):
    """
    (KL)
    Evaluates the Hill Function from an array of p. The form of the function is 
    x**p[0] / (p[1]**p[0] + x**p[0])
    """

    a, k = p
    return x ** a / (k ** a + x ** a)




"""
The following will plot the functions on a linear, semilog, and loglog scales. 
"""

# #############
def exp_decay_multiplot(p,x):
    """ 
    (GC)
    Plots the exponential decay function on linear, semilogy, semilogx,
    and loglog scale. 
    """
    #run the exponential function  
    exp = exp_decay(p,x)
    
    #plot the function as a figure

    plt.suptitle(r'$y = a + b\mathrm{e}^{-x / \lambda}$', fontsize=22) 
    #define the subplots
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(222)
    ax4 = plt.subplot(224)
   
    #plot linear scale
    ax1.plot(x, exp, 'g', alpha=0.5)
    ax1.set_ylabel(r'$y$')
    ax1.set_title(r'Linear', fontsize=10)
    ax1.grid(True)

     
    #plot semilogy scale
    ax2.semilogy(x, exp, 'g', alpha=0.5)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_title(r'Log $y$', fontsize=10)
    ax2.grid(True)
    
    #plot semilogx scale
    ax3.semilogx(x, exp, 'g', alpha=0.5)
    ax3.set_title(r'Log $x$', fontsize=10)
    ax3.grid(True)


    #plot loglog scale 
    ax4.loglog(x, exp, 'g', alpha=0.5)
    ax4.set_xlabel(r'$x$')
    ax4.set_title(r'Log $x$ Log $y$', fontsize=10)
    ax4.grid(True)

    #Make it look nice. 
    plt.tight_layout(pad=0.4, h_pad=1.0)
    plt.subplots_adjust(top=0.85) #leaves space for title.  
    
    #Save the figure
    plt.savefig('KEG_exp_multiplot.pdf', bbox_inches='tight', dpi=300,\
            transparent=True)

#Generate the array and domain
p = np.array([1, 100, 10])
x = np.linspace(0, 100, 1000)


#Call the function
exp_decay_multiplot(p,x)

plt.clf()
# ###############
def cauchy_multiplot(p, x):
    """
    (EB)
    Plots the Cauchy distribution on linear, semilogx, semilogy, and
    loglog scales.
    """
    #Evaulate the Cauchy Dist.   
    cauchy_eval = cauchy(p, x)
    
    #Define the figure
    plt.suptitle(r'$y = \frac{\beta}{\pi(\beta^2 + (x-\alpha)^2)}$', \
            fontsize=22)
 
   
    #define the subplots
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(222)
    ax4 = plt.subplot(224)
   
    #plot linear scale
    ax1.plot(x, cauchy_eval, 'b', alpha=0.5)
    ax1.set_title(r'Linear', fontsize=10)
    ax1.set_ylabel(r'$y$')
    ax1.grid(True)

     
    #plot semilogy scale
    ax2.semilogy(x, cauchy_eval, 'b', alpha=0.5)
    ax2.set_title(r'Log $y$', fontsize=10)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.grid(True)
    
    #plot semilogx scale
    ax3.semilogx(x, cauchy_eval, 'b', alpha=0.5)
    ax3.set_title(r'Log $x$', fontsize=10)
    ax3.grid(True)


    #plot loglog scale 
    ax4.loglog(x, cauchy_eval, 'b', alpha=0.5)
    ax4.set_title(r'Log $x$ Log $y$', fontsize=10)
    ax4.set_xlabel(r'$x$')
    ax4.grid(True)


    #make it look nice. 
    plt.tight_layout(pad=0.4, h_pad=1.0)
    plt.subplots_adjust(top=0.85) #leaves space for title.  
    
    #save the figure
    plt.savefig('KEG_cauchy_multiplot.pdf', bbox_inches='tight', dpi=300,\
            transparent=True)


#Generate the domain and domain and array
x = np.linspace(-10, 10, 1000)
p = np.array([1.0, 0])

#Call the function
cauchy_multiplot(p,x)


plt.clf()
# ################
def hill_multiplot(p, x):
    """
    (KL)

    Plots the Hill function with given parameters in semilogx, semilogy, linear, 
    and loglog.
    """
    #Evaluate the hill func. 
    hill_eval = hill(p,x)

    #Define the figure 
    plt.suptitle(r'$y= \frac{x^\alpha}{k^\alpha + x^\alpha}$', fontsize=22)

    #define the subplots
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(222)
    ax4 = plt.subplot(224)
   
    #plot linear scale
    ax1.plot(x, hill_eval, 'k', alpha=0.5)
    ax1.set_title(r'Linear', fontsize=10)
    ax1.set_ylabel(r'$y$')
    ax1.grid(True)

     
    #plot semilogy scale
    ax2.semilogy(x, hill_eval, 'k', alpha=0.5)
    ax2.set_title(r'Log $y$', fontsize=10)
    ax2.set_ylabel(r'$y$')
    ax2.set_xlabel(r'$x$')
    ax2.grid(True)
    
    #plot semilogx scale
    ax3.semilogx(x, hill_eval, 'k', alpha=0.5)
    ax3.set_title(r'Log $x$', fontsize=10)
    ax3.grid(True)


    #plot loglog scale 
    ax4.loglog(x, hill_eval, 'k', alpha=0.5)
    ax4.set_title(r'Log $x$ Log $y$', fontsize=10)
    ax4.set_xlabel(r'$x$')
    ax4.grid(True)


    #make it look nice. 
    plt.tight_layout(pad=0.4, h_pad=1.0)
    plt.subplots_adjust(top=0.85) #Leaves space at top for title.  
    
    #save the figure
    plt.savefig('KEG_hill_multiplot.pdf', bbox_inches='tight', dpi=300,\
            transparent=True)

#Generate the domain and array
x = np.linspace(0, 100, 1000)
p = np.array([2.0, 3.0])

#Call the function
hill_multiplot(p,x)

plt.clf()

"""
The following functions plot the exponential, Cauchy, and Hill functions with 
an array of values for the relevant variables. The best scaling was chosen for
each function (see accompanying document).
"""
# ##################
def exp_var(p, x):
    """
    (GC, EB, KL) 
    This function plots the exponential decay function with varying background
    (a) coefficient (b), and lambda (lam) by two orders of magnitude 
    respectively.
    """
    a, b, lam = p
    if lam == 0:
        raise ValueError("You can't divide by zero, stupid.")
    #Varying background signal
    
    y1 = (a / 10.0) + b * np.exp(-x / lam)
    y2 = a + b * np.exp(-x / lam)
    y3 = (a * 10.0) + b * np.exp(-x / lam)

    #Varying the coefficient
    y4 = a + (b / 10.0) * np.exp(-x / lam)
    y5 = a + (b * 10) * np.exp(-x / lam)
   
    #Varying lambda 
    y6 = a + b * np.exp(-x / (lam / 10))
    y7 = a + b * np.exp(-x / (lam * 10))

    #plotting the background variations. 
    plt.subplot(221)
    plt.semilogy(x, y1, 'g', alpha=0.5)
    plt.semilogy(x, y2, 'b', alpha=0.5) 
    plt.semilogy(x, y3, 'k', alpha=0.5)

   
    #Labeling the plot
    plt.legend([r'$a$ =%s' %(a / 10), r'$a$=%s' %a, r'$a$ =%s' %(a * 10)],\
            loc='upper right', fontsize=10)
    plt.title(r'Varying $a$')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')

    #plotting the coefficient variations. 
    plt.subplot(222)
    plt.semilogy(x, y4, 'g', alpha=0.5)
    plt.semilogy(x, y2, 'b', alpha=0.5)
    plt.semilogy(x, y5, 'k', alpha=0.5)

    #Labeling the plot.
    plt.yticks([])
    plt.xlabel(r'$x$')
    plt.title(r'Varying $b$')
    plt.legend([r'$b$ =%s' %(b / 10) , r'$b$ = %s' %b, r'$b$ =%s' %(b * 10)],\
            loc='upper right', fontsize=10)
    
    #Plotting the lambda variations
    plt.subplot(212)
    plt.semilogy(x, y6, 'g', alpha=0.5)
    plt.semilogy(x, y2, 'b', alpha=0.5)
    plt.semilogy(x, y7, 'k', alpha=0.5)

    #Labeling the plot 
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.ylim(0, (1.50 * (a + b)))
    plt.title(r'Varying $\lambda$')
    plt.legend([r'$\lambda$ =%s' %(lam / 10) , r'$\lambda$ = %s' %lam,\
            r'$\lambda$=%s' %(lam * 10)], loc='lower right', fontsize=10)
 
    #save the figures as a pdf. 
    plt.tight_layout()
    plt.savefig('KEG_exp_var.pdf', bbox_inches='tight', dpi=300,\
            transparent=True)

plt.clf()
#Generate teh domain and array
x = np.linspace(0, 1000, 1000)
p = np.array([100, 1000, 100])

#Call the function
exp_var(p, x)

# ##################
def cauchy_var(p,x):
    """
    (GC, EB, KL)
    This function plots the Cauchy distribution with varying beta and alpha 
    values by a factor of two or adding/subtracting two respectively.
    """
    #Define the values of beta and alpha as components of the array
    b, a = p

    #Vary beta
    y1 = (b / 2) / (np.pi * ((b / 2)**2 + (x - a)**2))
    y2 =  b / (np.pi * (b**2 + (x - a)**2))
    y3 = (b * 2) / (np.pi * ((b * 2)**2 + (x - a)**2 ))

    #Vary alpha
    y4 = b / (np.pi * (b**2 + (x - (a -2))**2))
    y5 = b / (np.pi * (b**2 + (x - (a + 2))**2))
    
    #Plotting the beta variations
    plt.subplot(121)
    plt.plot(x, y1, 'g', alpha=0.5)
    plt.plot(x, y2, 'b', alpha=0.5)
    plt.plot(x, y3, 'k', alpha=0.5)

    #Label the plot
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Varying $\beta$')
    plt.legend([r'$\beta$ = %s' %(b / 2), r'$\beta$ = %s' %b, r'$\beta$ = %s'\
            %(b * 2)], loc='upper left', fontsize=10)

    #Plotting the alpha variations
    plt.subplot(122)
    plt.plot(x, y4, 'g', alpha=0.5)
    plt.plot(x, y2, 'b', alpha=0.5)
    plt.plot(x, y5, 'k', alpha=0.5)
    
    plt.xlabel(r'$x$')
    plt.title(r'Varying $\alpha$')
    plt.legend([r'$\alpha$ = %s' %(a - 2), r'$\alpha$ = %s' %a, r'$\alpha$ =%s'\
            %(a + 2)], loc='upper left', fontsize=10)
    #Make it look nice. 
    plt.tight_layout()
    
    #Save the plots.
    plt.tight_layout()
    plt.savefig('KEG_cauchy_var.pdf', bbox_inches='tight', transparent=True,\
            dpi=300)


#Generate the domain and array
x = np.linspace(-10, 10, 1000)
p = np.array([1.0, 0.0])

#Call the function
cauchy_var(p,x)

plt.clf()

# ##################
def hill_var(p, x):
    """
    (GC, EB, KL)
    this function plots the hill function varying alpha and k by an order of
    magnitude and a factor of four respectively. 
    """
    #Define a and k from the provided array
    a, k = p
    #Varying alpha 
    y1 = x**(a/4) / (k**(a/4) + x**(a/4))
    y2 = x**a / (k**a + x**a) 
    y3 = x**(a*4) / (k**(a*4) + x**(a*4))

    #Varying k
    y4 = x**a / ((k/10)**a + x**a)
    y5= x**a / ((k * 10)**a + x**a)
    
    #Plotting the alpha variations.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.set_xlabel(r'$x$')
    ax2.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')

    ax1.semilogx(x, y1, 'g', alpha=0.5)
    ax1.semilogx(x, y2, 'b', alpha=0.5) 
    ax1.semilogx(x, y3, 'k', alpha=0.5)
    ax1.legend([r'$\alpha$ = %s' %(a / 4), r'$\alpha$ = %s' %a,\
            r'$\alpha$ = %s' %(a * 4)], loc='upper left', fontsize=10)
    ax1.set_title(r'Varying $\alpha$')
    
    #Plotting the k variations. 
    ax2.semilogx(x, y4, 'g', alpha=0.5)
    ax2.semilogx(x, y2, 'b', alpha=0.5)
    ax2.semilogx(x, y5, 'k', alpha=0.5)
   
    ax2.set_title(r'Varying $k$')
    ax2.legend([r'$k$ = %s' %(k / 10), r'$k$ = %s' %k, r'$k$ = %s' %(k * 10)],\
            loc='upper left', fontsize=10)
    #Make it look nice.
    plt.tight_layout()
    #Save the figures as a pdf. 
    plt.savefig('KEG_hill_var.pdf', bbox_inches='tight', dpi=300,\
            transparent=True)



#Generate the array and domain
x = np.linspace(0, 100, 1000)
p = np.array([2.0, 3.0])

#Call the function
hill_var(p, x)

plt.clf()


# #######################
# Code for problem 1.3  #
# #######################
"""
The following code generates the figures and analysis for problem 1.3
All parts will involve using pandas to read a csv file. These parameters are
addressed before any other data processing.
"""

#Use pd.read_csv to read the data and store in a dataframe
df = pd.read_csv('gardner_et_al_2011_time_to_catastrophe_dic.csv', comment='#')

#Rename columns for easier Use
df.columns=['labeled', 'unlabeled']



def mt_histogram(p, z):
    """
    (EB, KL, GC) 
    This will generate a histogram given an array of two data sets and the
    number of bins.
    """
    #Separate the array into two data sets 
    a, b = p

    #Normalizes the data sets. This is needed because each set has a different
    #number of observations.
    weights_a = np.ones_like(a) / len(a)
    weights_b = np.ones_like(b) / len(b)

    #Generate and plot the histogram.
    n, bin_edges, patches = plt.hist((a, b), weights=(weights_a, weights_b),
            bins=z, alpha=0.75, color=('green', 'grey'))
    plt.grid(True, axis='y')
    plt.xticks(np.arange(0,2000, 250))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency of Catastrophe')
    plt.legend(['Labeled', 'Unlabeled'])
    
    #Save the figure
    plt.savefig('KEG_catastrophe_histogram.pdf', bbox_inches='tight',
            transparent=True, dpi=300)

#Generate the array and set the number of bins
p = np.array([df.labeled, df.unlabeled.dropna()]) #dropna removes NaN vaules.
z = np.sqrt(len(df.labeled)) #Bin number determination via square root rule

#Call the function and plot
mt_histogram(p,z)

plt.clf()


def mt_cumulative_hist(p,z):
    """
    (GC, EB, KL)
    This function plots the cumulative histogram of collapse frequency using the
    same binning as used in part 1.2.c 
    """
    a, b = p

    #Normalizes the data sets. This is needed because each set has a different
    #number of observations.
    weights_a = np.ones_like(a) / len(a)
    weights_b = np.ones_like(b) / len(b)

    #Generate the histogram.
    (n,m), bin_edges, patches = plt.hist((a, b), weights=(weights_a, weights_b),
            histtype='bar', bins=z, alpha=0.1, color=('green', 'grey'), cumulative=True)
    #Compute the bar midpoint. 
    x = (bin_edges[1:]+bin_edges[:-1])/2

    #Plot the histogram as points. 
    plt.plot(x, n, 'g o', alpha=0.75, ms=10); plt.plot(x, m, 'ko', alpha=0.75,
            ms=10)
    plt.grid(True, axis='y')
    plt.xticks(np.arange(0,2000, 250))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency of Catastrophe')
    plt.legend(['Labeled', 'Unlabeled'], loc='lower right', numpoints=1)

    #Save the figure
    plt.savefig('KEG_catastrophe_cumulative_histogram.pdf', bbox_inches='tight',
            transparent=True, dpi=300)
  
#Generate the array to plot and set the bin number
p = np.array([df.labeled, df.unlabeled.dropna()])
z = np.sqrt(len(df.labeled)) #Bin number determination via square root rule

#Call the function to plot
mt_cumulative_hist(p,z)

plt.clf()




"""
The code below generates a figure similar to that of 2A found in the paper. 
It functions as a cumulative sum of catastrophe events as a function of time. 
"""

#Generate copies of the labeled and unlabeled events and sort them. 

label_sort = df.labeled.copy()
label_sort.sort()
x_labeled = label_sort

unlabel_sort = df.unlabeled.copy() 
unlabel_sort.sort()
x_unlabeled= unlabel_sort.dropna()

#Generate the y values using np.arange. This will serve as the cumulative sum
y_labeled = np.arange(1, len(x_labeled) + 1) / len(x_labeled)
y_unlabeled = np.arange(1, len(x_unlabeled) + 1) / len(x_unlabeled)

# ######################
def mt_cumulative_nobin(p,x):
    """
    (GC, EB, KL)
    This function plots two sets of x and y data given as arrays as a cumulative
    histogram with no binning. 
    """
    #Define the data sets from the arrays

    a, b = p #x values for labeled and unlabeled
    c, d = x #y values for labeled and Unlabeled

    #Generate the plot
    fig, ax = plt.subplots()

    plt.plot(a, c, 'go', alpha=0.75)
    plt.plot(b, d, 'ko', alpha=0.75)
    plt.legend([r'Labeled', r'Unlabeled'], loc='lower right', fontsize=10,\
            numpoints=1)

    #Label the plots.
    ax.set_xlabel(r'Time to Catastrophe (s)')
    ax.set_ylabel(r'Cumulative Distribution')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.grid(True)
    plt.ylim(0,1.1)

    #Make it look nice.
    plt.tight_layout()

    #Save the figure.
    plt.savefig('KEG_cumulative_dist.pdf', bbox_inches='tight',\
            transparent=True, dpi=300)

#Generate the arrays for the plot.
p = np.array([x_labeled,x_unlabeled])
x = np.array([y_labeled, y_unlabeled])
#Call the function to plot
mt_cumulative_nobin(p,x)

