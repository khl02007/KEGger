def pretty_activity_plot(ax, x, y, df, xlabel='time (hr)', 
                         ylabel='activity (sec / min)', lw=0.25):
    """
    Makes a pretty plot of sleep traces.  Generates the plot on axes ax,
    and then returns the updated ax.
    """
    # Generate plot (black line if single trace, with colors otherwise)
    if len(y.shape) == 1 or (np.array(y.shape) == 1).any():
        ax.plot(x, y, 'k-', lw=lw)

        # y_max is maximal y value encountered
        y_max = y.max()
    else:
        ax.plot(x, y, '-', lw=lw)

        # Do it this way for compatibility w/ DataFrames and 2d NumPy arrays
        y_max = y.max(axis=0).max()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((0.0, y_max))

    # We can overlay day and night.  We'll make night shaded
    ax.fill_between(df.zeit, 0.0, 50*y_max, where=~df.light, 
                    color='lightgray',  alpha=0.5, zorder=0)
    
    return ax
