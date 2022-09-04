import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(mappable, dummy=False):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if dummy:
        cax.set_frame_on(False)
        cax.axes.get_yaxis().set_visible(False)
        cax.axes.get_xaxis().set_visible(False)
        cbar = None
    else:
        cbar = fig.colorbar(mappable, cax=cax)
        # plt.clim(0.0, 0.02)
        # cbar.set_clim()
        plt.sca(last_axes)
    return cbar