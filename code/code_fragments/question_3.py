from __future__ import division, absolute_import, \
print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Use pd.read_csv to read the data and store in a DataFrame

df = pd.read_csv('gardner_et_al_2011_time_to_catastrophe_dic.csv', comment='#')

#Rename columns for easier Use

df.columns=['labeled', 'unlabeled']


#Get rid of the 'NaN' data

#Find and remove all 'NaN'
#unlabeled_numbers = df.unlabeled.dropna()

#a = df.labeled
#b = df.unlabeled[0:94]
#labeled_bin_no= np.sqrt(211)
#unlabeled_bin_no= np.sqrt(95)

#Make a histogram plot

def mt_histogram(p, z):
    """
    This will generate a histogram given an array of two data sets and the
    number of bins.
    """
    
    a, b = p

    n, bin_edges, patches = plt.hist((a), bins=1, normed=True, stacked=z, alpha=0.75, \
    histtype='bar', color=('green'), rwidth=1)
    plt.grid(True, axis='y')
    plt.xlabel('Time to catastrophe (s)')
    plt.ylabel('Number of observations')
    plt.legend(['Labeled', 'Unlabeled'])




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

#Generate the plot of the cumulative sum. 
fig, ax = plt.subplots()
plt.plot(x_labeled, y_labeled, 'g o', alpha=0.5)
plt.plot(x_unlabeled, y_unlabeled, 'k o', alpha=0.5)
plt.legend([r'Labeled',r'Unlabeled'], loc='lower right', fontsize=10,
        numpoints=1)
plt.ylabel(r'Cumulative Distribution')
plt.xlabel(r'Time to Catastrophe (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.ylim(0,1.1)

plt.savefig('KEG_cumulative_dist.pdf', bbox_inches='tight', transparent=True,\
        dpi=300)
