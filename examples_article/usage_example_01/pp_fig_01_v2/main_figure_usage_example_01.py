import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def set_legend(ax, lines, loc=0, ncol=1):
    """set legend

    Function generating a custom legend, see [1] for more options

    Refs:
      [1] https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html

    Args:
      ax (object): figure part for which the legend is intended
      lines (list): list of  Line2D objects
    """
    # -- extract labels from lines
    labels = [x.get_label() for x in lines]
    # -- customize legend 
    lg = ax.legend(lines,                # list of Line2D objects
              labels,               # labels 
              title = '',           # title shown on top of legend 
              loc = loc,              # location of the legend
              ncol = ncol,             # number of columns
              labelspacing = 0.3,   # vertical space between handles in font-size units
              borderpad = 0.3,      # distance to legend border in font-size units
              handletextpad = 0.3,  # distance between handle and label in font-size units
              handlelength = 1.5,   # length of handle in font-size units
              frameon = False # remove background patch
              )
    return lg


def save_figure(fig_name = 'test', fig_format = None):
    """ save figure

    Function that saves figure or shows interactive plot

    Note:
    - if no valid option is provided, an interactive plot is shown

    Args:
      fig_format (str): format to save figure in (options: png, pdf, svg)
      fig_name (str): name for figure (default: 'test')
    """
    if fig_format == 'png':
        plt.savefig(fig_name+'.png', format='png', dpi=600)
    elif fig_format == 'pdf':
        plt.savefig(fig_name+'.pdf', format='pdf', dpi=600)
    elif fig_format == 'svg':
        plt.savefig(fig_name+'.svg', format='svg', dpi=600)
    else:
        plt.show()



def set_style():
    """set figure style

    Function that customizes the default style to be conform with the Physical
    Review style and notation guide [1]. For instructions on how to set the
    default style using style sheets see [2].

    Notes:
    - main font size is chosen as 8pt, matching the fontsize of figure captions
    - fontsize of legends and auxiliary text labels are set to 6pt, exceeding
      the minimally required pointsize of 4.25 pt. (1.5 mm)
    - default rc (rc = "run commands", i.e. startup information) settings are
      changed dynamically
    - the custom font-scheme 'type2' depends on your latex installation and
      is not guaranteed to run on your specific system

    Refs:
      [1] https://journals.aps.org/prl/authors
      [2] https://matplotlib.org/3.3.1/tutorials/introductory/customizing.html
    """

    fig_width_1col = 3.4        # figure width in inch
    fig_width_2col = 7.0        # figure width in inch
    fig_aspect_ratio = 0.55     # width to height aspect ratio
    font_size = 7               # font size in pt
    font_size_small = 5         # font size in pt

    fig_width = fig_width_1col
    fig_height = fig_aspect_ratio*fig_width_1col

    params = {
        'figure.figsize': (fig_width,fig_height),
        'legend.fontsize': 6,
        'legend.frameon': False,
        'axes.labelsize': 7,
        'axes.linewidth': 0.8,
        'xtick.labelsize' :7,
        'ytick.labelsize': 7,
        'mathtext.fontset': 'stixsans',
        'mathtext.rm': 'serif',
        'mathtext.bf': 'serif:bold',
        'mathtext.it': 'serif:italic',
        'mathtext.sf': 'sans\\-serif',
        'font.size':  7,
        'font.family': 'serif',
        'font.serif': "Helvetica",
        'grid.linestyle': ':',
        'grid.linewidth': 0.75,
        'grid.color': 'gray',
             }

    mpl.rcParams.update(params)




def figure(fig_basename):

    set_style()

    # (2) SET FIGURE LAYOUT
    fig = plt.figure()
    plt.subplots_adjust(left = 0.15, bottom = 0.17, right = 0.96, top = 0.96,
    wspace = .5, hspace = 2.4)

    gs00 = GridSpec(nrows = 1, ncols = 1)
    gsA = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs00[0:5,0], wspace=0.07, hspace=0.07)
    axA1 = fig.add_subplot(gsA[0, 0])

    col1 = 'k' # '#AF58BA' #'#0f63ff'
    col2 = 'gray' #'#00CD6C'    # '#ffab0f' #'#b45c1f'
    col12 = 'k' #'#AF58BA'  # '#0f63ff'
    col22 = 'gray' #'#00CD6C' # '#ffab0f' #'#b45c1f'
    lw2 = 1.

    # -- RK4 RESULTS 

    f_name = '../res_SySSM_RK2.dat'
    dat = np.loadtxt(f_name)
    dz = dat[:,0]
    err = dat[:,1]
    l2 = axA1.plot(dz, err, color=col1, marker='s', markersize=3., markerfacecolor=col12, linewidth=1., label=r'SySSM-RK2')

    f_name = '../res_IFM_RK4.dat'
    dat = np.loadtxt(f_name)
    dz = dat[:,0]
    err = dat[:,1]
    l3 = axA1.plot(dz, err, color=col1, marker='o', markersize=3., markerfacecolor=col12, linewidth=1., label=r'IFM-RK4IP')

    # -- RK2 RESULTS 

    f_name = '../res_SiSSM_RK2.dat'
    dat = np.loadtxt(f_name)
    dz = dat[:,0]
    err = dat[:,1]
    l4 = axA1.plot(dz, err, color=col1, marker='D', markersize=3., markerfacecolor=col1, linewidth=1, label=r'SiSSM-RK2')
    #l4 = axA1.plot(dz, err, color=col2, marker='D', markersize=2., markerfacecolor=col22, linewidth=lw2, dashes=[2,2], label=r'SiSSM')

    # -- LEM RESULTS 

    col12=col22='white'

    f_name = '../res_LEM_RK2.dat'
    dat = np.loadtxt(f_name)
    dz = dat[:,0]
    err = dat[:,1]
    l6 = axA1.plot(dz, err, color=col1, marker='^', markersize=3., markerfacecolor=col12, linewidth=1., dashes=[2,2], mew = 1., label=r'LEM-RK2')

    f_name = '../res_LEM_RK4.dat'
    dat = np.loadtxt(f_name)
    dz = dat[:,0]
    err = dat[:,1]
    l7 = axA1.plot(dz, err, color=col1, marker='^', markersize=3., markerfacecolor=col12, linewidth=1., mew= 1., label=r'LEM-RK4')

    set_legend(axA1, l4+l2+l3+l6+l7, loc=(0.7,0.06), ncol=2)
    #lg1 = set_legend(axA1, l4+l5, loc=(0.4,0.06), ncol=1)
    #lg2 = set_legend(axA1, l1+l2+l3, loc=(0.7,0.06), ncol=1)
    #plt.gca().add_artist(lg1)
    #plt.gca().add_artist(lg2)
    axA1.grid(True)


    line = lambda a, b, x: a*x**b
    dz_ = np.linspace(2e-3,9e-3,5)
    axA1.plot(dz_, line(0.15, 1, dz_), linewidth=0.75, color='darkgray')
    axA1.plot(dz_, line(0.15, 2, dz_), linewidth=0.75, color='darkgray')
    axA1.plot(dz_, line(0.02, 3, dz_), linewidth=0.75, color='darkgray')
    axA1.plot(dz_, line(0.10, 4, dz_), linewidth=0.75, color='darkgray')

    dz_ = np.linspace(0.9e-4,2e-3,5)
    axA1.plot(dz_, line(4e-18, -1, dz_), linewidth=0.75, color='darkgray')

    axA1.text( 0.35, 0.85, r'$\propto \Delta z$', transform=axA1.transAxes)
    axA1.text( 0.35, 0.66, r'$\propto \Delta z^2$', transform=axA1.transAxes)
    axA1.text( 0.35, 0.41, r'$\propto \Delta z^3$', transform=axA1.transAxes)
    axA1.text( 0.35, 0.28, r'$\propto \Delta z^4$', transform=axA1.transAxes)
    axA1.text( 0.08, 0.04, r'$\propto \Delta z^{-1}$', transform=axA1.transAxes)

    dz_lim = (7e-5,0.13)
    dz_ticks = (1e-4, 1e-3, 1e-2, 1e-1)
    axA1.tick_params(axis='x', length=2., pad=2, top=False)
    axA1.set_xscale('log')
    axA1.set_xlim(dz_lim)
    axA1.set_xticks(dz_ticks)
    axA1.set_xlabel(r"Step size $\Delta z~\mathrm{(\mu m)}$")

    err_lim = (0.1e-15,1e-1)
    err_ticks = (1e-14,1e-10,1e-6,1e-2)
    #err_ticks = (1e-16,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2)
    axA1.tick_params(axis='y', length=2., pad=2, top=False)
    axA1.set_yscale('log')
    axA1.set_ylim(err_lim)
    axA1.set_yticks(err_ticks)
    axA1.set_ylabel(r"Average RMS error $\epsilon$")
    #axA1.set_ylabel(r"Global error $\epsilon$")

    save_figure(fig_name = fig_basename, fig_format = 'png')



def main():
    file_name = 'fig01'
    figure(file_name)


if __name__ == "__main__":
    main()
