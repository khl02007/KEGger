plt.subplots(nrows=2, ncols=2)


"""
k scales the x axis. 

plot as x/k. 


no binning for final question. 

sort x labeled

np.arange(1, len(labeled)+1, 1)


"""


df.labeled.copy()

labeled.sort()


"""

for labeling 

plt.text(xval, yval, s)

plt.text(0.6, 0.6, r'$\mathrm{e}^{\sin x}$', transform 



fig, ax = plt.subplots()
ax.plot(x,y)
ax.text(0.6, 0.6, r'$\mathrm{E}^{\sin x}$', fontsize=24, transform=ax.transAxes)
