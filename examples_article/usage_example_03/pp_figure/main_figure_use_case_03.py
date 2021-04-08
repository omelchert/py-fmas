"""
Author: O. Melchert
Date: 2020-09-09
"""
import sys
import os
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import h5py

__author__ = 'Oliver Melchert'
__date__ = '2020-09-09'


# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft

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
    ax.legend(lines,                # list of Line2D objects
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


def custom_colormap():
    # -- CREATE CUSTOM COLORMAP
    from matplotlib.colors import ListedColormap
    cmap_base = mpl.cm.jet(np.arange(256))
    blank = np.ones((25,4))
    for i in range(3):
       blank[:,i] = np.linspace(1,cmap_base[0,i], blank.shape[0])
    my_cmap = ListedColormap(np.vstack((blank, cmap_base )))
    return my_cmap

class Figure():

    def __init__(self, aspect_ratio = 1.0, fig_format = 'png', fig_basename = 'fig_test'):
        self.fig_format = fig_format
        self.fig_basename = fig_basename
        self.fig = None
        self.set_style(aspect_ratio)

    def set_style(self, aspect_ratio = 1.0):

        fig_width = 3.4 # (inch)
        fig_height = aspect_ratio*fig_width

        params = {
            'figure.figsize': (fig_width,fig_height),
            'legend.fontsize': 6,
            'legend.frameon': False,
            'axes.labelsize': 7,
            'axes.linewidth': 1.,
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
        }
        mpl.rcParams.update(params)


    def save(self):
        fig_format, fig_name = self.fig_format, self.fig_basename
        if fig_format == 'png':
            plt.savefig(fig_name+'.png', format='png', dpi=600)
        elif fig_format == 'pdf':
            plt.savefig(fig_name+'.pdf', format='pdf', dpi=600)
        elif fig_format == 'svg':
            plt.savefig(fig_name+'.svg', format='svg', dpi=600)
        else:
            plt.show()


    def set_subfig_label(self,ax, label, loc=1):
            pos = ax.get_position()

            if loc==1:
                self.fig.text(pos.x0, pos.y1, label ,color='white',
                    backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                    boxstyle='square,pad=0.1'), verticalalignment='top' )

            elif loc==2:
                self.fig.text(pos.x0, pos.y0, label ,color='white',
                    backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                    boxstyle='square,pad=0.1'), verticalalignment='bottom' )

            else:
                print("check label position")
                exit()


    def set_layout(self):

        fig = plt.figure()
        self.fig = fig

        plt.subplots_adjust(left = 0.13, bottom = 0.08, right = 0.88, top = 0.93, wspace = .5, hspace = 2.4)



        gs00 = GridSpec(nrows = 11, ncols = 1)
        gsA = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs00[0:5,0], wspace=0.07, hspace=0.07)
        axA1 = fig.add_subplot(gsA[1:, 0:2])
        axA2 = fig.add_subplot(gsA[1:, 2])
        self.subfig_1 = [axA1, axA2]

        gsC = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs00[5:8,0], wspace=0.1, hspace=1.)
        axC = fig.add_subplot(gsC[0:4,0])
        self.subfig_2 = axC

        gsD = GridSpecFromSubplotSpec(5, 2, subplot_spec=gs00[8:,0], wspace=0.07, hspace=1.)
        axD1 = fig.add_subplot(gsD[0:4,0])
        axD2 = fig.add_subplot(gsD[0:4,1])
        self.subfig_3 = [axD1,axD2]

    def set_subfig_01(self):
        axA1, axA2 = self.subfig_1

        z,t,w,utz, Cp = fetch_data('../out_file_HR.h5')
        t = t*1e-3 # fs -> ps
        z = z*1e-6 # micron -> m

        self.t = t
        self.z = z
        self.w = w
        self.utz = utz

        t_lim = (-4.3,1.3)
        #t_lim = (-8.,8.)
        t_ticks = (-4,-3,-2,-1,0,1)
        z_lim = (0,4.)
        z_ticks  = (0,1,2,3,4)
        w_lim = (1.25,2.25)
        w_ticks = (1.3,1.6,1.9,2.2)

        _norm = lambda x: x/x[0].max()
        _truncate = lambda x: np.where(x>1.e-20,x,1.e-20)
        _dB = lambda x: np.where(x>1e-20,10.*np.log10(x),10*np.log10(1e-20))

        def set_colorbar(fig, img, ax, label='text', dw=0.):
            # -- EXTRACT POSITION INFORMATION FOR COLORBAR PLACEMENT
            refPos = ax.get_position()
            x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
            # -- SET NEW AXES AS REFERENCE FOR COLORBAR
            colorbar_axis = fig.add_axes([x0+dw, y0 + h + .03*h, w-dw, 0.04*h])
            # -- SET CUSTOM COLORBAR
            colorbar = fig.colorbar(img,        # image described by colorbar
                    cax = colorbar_axis,        # reference axex
                    orientation = 'horizontal', # colorbar orientation
                    extend = 'both'             # ends with out-of range values
                    )
            colorbar.ax.tick_params(
                    color = 'k',                # tick color 
                    labelcolor = 'k',           # label color
                    bottom = False,             # no ticks at bottom
                    labelbottom = False,        # no labels at bottom
                    labeltop = True,            # labels on top
                    top = True,                 # ticks on top
                    direction = 'out',          # place ticks outside
                    length = 2,                 # tick length in pts. 
                    labelsize = 5.,             # tick font in pts.
                    pad = 1.                    # tick-to-label distance in pts.
                    )
            fig.text(x0, y0+h+0.03*h, label, horizontalalignment='left', verticalalignment='bottom', size=6)
            return colorbar

        my_cmap = custom_colormap()

        # -- PROPAGATION CHARACERISTICS 
        # ... BOTTOM PLOT

        Itz = _truncate(_norm(np.abs(utz)**2))

        img = axA1.pcolorfast(t, z, _dB(Itz[:-1,:-1]),
                              vmin=-40, vmax=0.,
                              cmap = my_cmap
                              )
        cb = set_colorbar(self.fig, img, axA1, label='$|\mathcal{E}|^2\,\mathrm{(dB)}$', dw=0.1)
        cb.set_ticks((-40,-30,-20,-10,0))
        #cb.ax.set_title('Intensity $|\mathcal{E}|^2\,\mathrm{(dB)}$', fontsize=7., y=2.5)

        axA1.tick_params(axis='x', length=2., pad=2, top=False)
        axA1.set_xlim(t_lim)
        axA1.set_xticks(t_ticks)
        axA1.set_xlabel(r"Time $\tau~\mathrm{(ps)}$")

        axA1.tick_params(axis='y', length=2., pad=2, top=False)
        axA1.set_ylim(z_lim)
        axA1.set_yticks(z_ticks)
        axA1.set_ylabel(r"Propagation distance $z~\mathrm{(m)}$")


        # -- FREQUENCY DOMAIN
        # ... BOTTOM PLOT
        uwz = FT(utz,axis=-1)
        Iwz_s = nfft.fftshift(_truncate(_norm(np.abs(uwz)**2)),axes=-1)
        w_s  = nfft.fftshift(w)

        w_s_mask = np.logical_and(w_s>w_lim[0], w_s<6) #w_lim[1])
        Iwz_s_f = Iwz_s[:,w_s_mask]
        w_s_f = w_s[w_s_mask]
        c0 = 0.29979
        #lam = 2*np.pi*c0/w_s_f
        _lam = lambda w: 2*np.pi*c0/w
        _w = lambda x: 2*np.pi*c0/x
        lam = _lam(w_s_f)

        img = axA2.pcolorfast(w_s, z, _dB(Iwz_s[:-1,:-1]),
                              vmin=-40, vmax=0.,
                              cmap = my_cmap
                              )
        cb = set_colorbar(self.fig, img, axA2,  label='$|\mathcal{E}_\omega|^2\,\mathrm{(dB)}$', dw=0.1)
        cb.set_ticks((-40,-20,0))

        w_ZDW = 1.7408
        axA2.axvline( w_ZDW, color='k', dashes=[2,2], linewidth=1)

        axA2.tick_params(axis='x', length=2., pad=2, top=False)

        lam_lim = (0.45,1.35)
        lam_ticks = (0.5,0.7,0.9,1.1,1.3)
        axA2.set_xlim(w_lim)
        axA2.set_xticks(w_ticks)
        #axA2.set_xticklabels( (1,2,r'$\omega_{\rm{Z}}$',3,4)  )
        axA2.set_xlabel(r"Ang. freq. $\omega~\mathrm{(rad/fs)}$")

        axA2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
        axA2.set_ylim(z_lim)
        axA2.set_yticks(z_ticks)


        self.set_subfig_label(axA1,'(b)',loc=1)
        self.set_subfig_label(axA2,'(c)',loc=1)

    def set_subfig_02(self):
        axD = self.subfig_2

        import fmas.propagation_constant

        beta_fun = fmas.propagation_constant.define_beta_fun_ESM()
        pc = fmas.propagation_constant.PropConst(beta_fun)

        # -- PROPERTIES OF THE PROPAGATION CONSTANT
        # ... GROUP VELOCITY DISPERSION 

        w_lim = (1.,3.)
        w_ticks = (1,1.5,1.7408,2.,2.5,3.)
        w = np.linspace(1,w_lim[1],200)

        b2 = pc.beta2(w)
        l1 = axD.plot(w, b2, color='blue', linewidth=1, label=r'$\beta_2$')

        w_ZDW = 1.7408
        axD.axvspan(w_ZDW, w_lim[1], color='lightgray')

        axD.tick_params(axis='x', length=2., pad=2, top=False)
        axD.set_xlim(w_lim)
        axD.set_xticks(w_ticks)
        axD.set_xticklabels( (1,1.5,r'$\omega_{\rm{Z}}$',2,2.5,3)  )
        axD.set_xlabel(r"Angular frequency $\omega~\mathrm{(rad/fs)}$")

        axD.tick_params(axis='y', length=2., pad=2, top=False, labelleft=True, left=True, colors='blue')
        axD.spines['left'].set_color('blue')
        axD.set_ylim(-0.15,0.05)
        axD.set_yticks((-0.15,-0.1,-0.05,0,0.05))
        axD.set_ylabel(r"GVD $\beta_2~\mathrm{(fs^2/\mu m)}$",color='blue')

        # ... GROUP DELAY 

        axD2 = axD.twinx()

        b1 = pc.beta1(w)
        l2 = axD2.plot(w, b1, color='red', linewidth=1, dashes=[3,1], label=r'$\beta_1$')

        set_legend(axD, l1,loc=3)
        set_legend(axD2, l2,loc=4)

        axD2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False, labelright=True, left=False, right=True, colors='red')
        axD2.set_ylim(4.89,4.93)
        axD2.set_yticks(( 4.89, 4.9, 4.91,4.92,4.93   ))
        axD2.set_ylabel(r"rGD $\beta_1~\mathrm{(fs/\mu m)}$",color='red')

        axD2.spines['left'].set_color('blue')
        axD2.spines['right'].set_color('red')

        self.set_subfig_label(axD,'(a)',loc=1)


    def set_subfig_03(self):
        ax1, ax2 = self.subfig_3

        _norm = lambda x: x/x[0].max()
        _truncate = lambda x: np.where(x>1.e-20,x,1.e-20)
        _dB = lambda x: np.where(x>1e-20,10.*np.log10(x),10*np.log10(1e-20))

        from fmas.tools import spectrogram

        my_cmap = custom_colormap()

        z, t, w, utz = self.z, self.t, self.w, self.utz

        t_min, t_max = -4.,2.5
        w_min, w_max = 1.25, 2.25

        v_min = -45
        s0 = 0.05

        z0_idx = np.argmin(np.abs(z-0.))
        Et = utz[z0_idx]
        t_delay, w_s, P_tw = spectrogram(t, w, Et, t_lim=(t_min, t_max), Nt=600, Nw=512, s0=s0)
        w_mask = np.logical_and(w_s>1.3,w_s<2.3)

        def set_colorbar_2(fig, img, ax, label='text', dw=0.):
            # -- EXTRACT POSITION INFORMATION FOR COLORBAR PLACEMENT
            refPos = ax.get_position()
            x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
            # -- SET NEW AXES AS REFERENCE FOR COLORBAR
            colorbar_axis = fig.add_axes([x0+w+0.03*w, y0, 0.05*w, 0.85*h])
            # -- SET CUSTOM COLORBAR
            colorbar = fig.colorbar(img,        # image described by colorbar
                    cax = colorbar_axis,        # reference axex
                    orientation = 'vertical', # colorbar orientation
                    extend = 'both'             # ends with out-of range values
                    )
            colorbar.ax.tick_params(
                    color = 'k',                # tick color 
                    labelcolor = 'k',           # label color
                    bottom = False,             # no ticks at bottom
                    labelbottom = False,        # no labels at bottom
                    labeltop = True,            # labels on top
                    top = True,                 # ticks on top
                    direction = 'out',          # place ticks outside
                    length = 2,                 # tick length in pts. 
                    labelsize = 5.,             # tick font in pts.
                    pad = 1.                    # tick-to-label distance in pts.
                    )
            fig.text(x0 +1.03*w, y0+h, label, horizontalalignment='left', verticalalignment='top', size=6)
            return colorbar

        P_tw = P_tw[w_mask,:]
        I = P_tw[:-1,:-1]/P_tw.max()
        img = ax1.pcolorfast(
            t_delay, w_s[w_mask], _dB(I),
            vmin=v_min,vmax=0,
            cmap=my_cmap)
        #cb = set_colorbar(self.fig, img, ax1,  label=r'$P_S\,\mathrm{(dB)}$', dw=0.15)

        ax1.set_xlim(t_min,t_max)
        ax1.set_xticks((-4,-2,0,2))
        ax1.set_ylim(w_min,w_max)
        ax1.set_yticks((1.3,1.6,1.9,2.2))
        ax1.tick_params(axis='y',length=2., direction='out')
        ax1.tick_params(axis='x',length=2., direction='out')
        ax1.set_xlabel(r"Time $\tau~\mathrm{(ps)}$")
        ax1.set_ylabel(r"Ang. freq. $\omega~\mathrm{(rad/fs)}$")


        t_min, t_max = -4.,2.5

        z0_idx = np.argmin(np.abs(z-2.2))
        Et = utz[z0_idx]
        t_delay, w_s, P_tw = spectrogram(t, w, Et, t_lim=(t_min, t_max), Nt=600, Nw=512, s0=s0)
        w_mask = np.logical_and(w_s>1.3,w_s<2.3)

        P_tw = P_tw[w_mask,:]
        I = P_tw[:-1,:-1]/P_tw.max()
        img = ax2.pcolorfast(
            t_delay, w_s[w_mask], _dB(I),
            vmin=v_min,vmax=0,
            cmap=my_cmap)
        cb = set_colorbar_2(self.fig, img, ax2,  label=r'$P_S\,\mathrm{(dB)}$', dw=0.15)
        cb.set_ticks((-40,-30,-20,-10,0))

        ax2.set_xlim(t_min,t_max)
        ax2.set_xticks((-4,-2,0,2))
        ax2.set_ylim(w_min,w_max)
        ax2.set_yticks((1.3,1.6,1.9,2.2))
        ax2.tick_params(axis='y',length=2., direction='out', labelleft=False)
        ax2.tick_params(axis='x',length=2., direction='out')
        ax2.set_xlabel(r"Time $\tau~\mathrm{(ps)}$")

        ax1.text(.025, 0.025, r'$z=0.0~\mathrm{m}$', horizontalalignment='left', verticalalignment='bottom', size=7, transform=ax1.transAxes)
        ax2.text(.025, 0.025, r'$z=2.2~\mathrm{m}$', horizontalalignment='left', verticalalignment='bottom', size=7, transform=ax2.transAxes)

        self.set_subfig_label(ax1,'(d)',loc=1)
        self.set_subfig_label(ax2,'(e)',loc=1)


def fetch_data(file_path):

    with h5py.File(file_path, "r") as f:
        utz = np.array(f['utz'])
        t = np.array(f['t'])
        w = np.array(f['w'])
        z = np.array(f['z'])
        Cp = np.array(f['Cp'])
    return z,t,w,utz, Cp


def main():

    myFig = Figure(aspect_ratio=1.6, fig_basename='fig_03', fig_format='png')
    myFig.set_layout()
    myFig.set_subfig_01()
    myFig.set_subfig_02()
    myFig.set_subfig_03()
    myFig.save()

if __name__ == '__main__':
    main()
