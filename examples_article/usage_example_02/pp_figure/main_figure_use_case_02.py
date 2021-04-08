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
        axA1 = fig.add_subplot(gsA[1:, 0])
        axA2 = fig.add_subplot(gsA[1:, 1:])
        axB1 = fig.add_subplot(gsA[0, 0])
        axB2 = fig.add_subplot(gsA[0, 1:])
        self.subfig_1 = [axA1, axA2, axB1, axB2]

        gsC = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs00[5:8,0], wspace=0.1, hspace=1.)
        axC = fig.add_subplot(gsC[0:4,0])
        self.subfig_2 = axC

        gsD = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs00[8:,0], wspace=0.1, hspace=1.)
        axD = fig.add_subplot(gsD[0:4,0])
        self.subfig_3 = axD


    def set_subfig_01(self):
        axA1, axA2, axB1, axB2 = self.subfig_1

        z,t,w,utz, dz_int, Cp = fetch_data('../res_IFM_SC_Nz4000.h5')
        t = t*1e-3 # fs -> ps
        z = z*1e-4 # micron -> cm

        t_lim = (-0.5,3.5)
        t_ticks = (0,1,2,3)
        z_lim = (0,14)
        z_ticks  = (0,4,8,12)
        w_lim = (1,4.2)
        w_ticks = (1,2,2.415,3,4)

        _norm = lambda x: x/x[0].max()
        _truncate = lambda x: np.where(x>1.e-20,x,1.e-20)
        _dB = lambda x: np.where(x>1e-20,10.*np.log10(x),10*np.log10(1e-20))

        def set_colorbar(fig, img, ax, ax2, label='text', dw=0.):
            # -- EXTRACT POSITION INFORMATION FOR COLORBAR PLACEMENT
            refPos = ax.get_position()
            x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
            h2 = ax2.get_position().height
            y2 = ax2.get_position().y0
            # -- SET NEW AXES AS REFERENCE FOR COLORBAR
            colorbar_axis = fig.add_axes([x0+dw, y2 + h2 + .03*h, w-dw, 0.04*h])
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
            fig.text(x0, y2+h2+0.03*h, label, horizontalalignment='left', verticalalignment='bottom', size=6)
            return colorbar

        my_cmap = custom_colormap()

        # -- PROPAGATION CHARACERISTICS 
        # ... BOTTOM PLOT

        Itz = _truncate(_norm(np.abs(utz)**2))

        img = axA1.pcolorfast(t, z, _dB(Itz[:-1,:-1]),
                              vmin=-40, vmax=0.,
                              cmap = my_cmap
                              )
        cb = set_colorbar(self.fig, img, axA1, axB1, label='$|\mathcal{E}|^2\,\mathrm{(dB)}$', dw=0.1)
        cb.set_ticks((-40,-20,0))
        #cb.ax.set_title('Intensity $|\mathcal{E}|^2\,\mathrm{(dB)}$', fontsize=7., y=2.5)

        axA1.tick_params(axis='x', length=2., pad=2, top=False)
        axA1.set_xlim(t_lim)
        axA1.set_xticks(t_ticks)
        axA1.set_xlabel(r"Time $\tau\,\mathrm{(ps)}$")

        axA1.tick_params(axis='y', length=2., pad=2, top=False)
        axA1.set_ylim(z_lim)
        axA1.set_yticks(z_ticks)
        axA1.set_ylabel(r"Propagation distance $z\,\mathrm{(cm)}$")

        # ... UPPER PLOT 
        #z_id = np.argmin(np.abs(z-14))
        #It = Itz[z_id]
        #axB1.plot(t, It/np.max(It), color='k', linewidth=.75)

        #axB1.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
        #axB1.set_xlim(t_lim)
        #axB1.set_xticks(t_ticks)

        #axB1.tick_params(axis='y', length=2., pad=2, top=False, labelleft=True, left=True)
        #axB1.set_ylim((0,1.1))
        #axB1.set_yticks((0,0.5,1))
        #axB1.set_ylabel(r"$|\mathcal{E}|^2\,\mathrm{(norm.)}$")

        # ... UPPER PLOT - real field 
        z_id = np.argmin(np.abs(z-14))
        Eopt = np.real(utz[z_id])
        axB1.plot(t, Eopt/np.max(Eopt), color='k', linewidth=.75)

        axB1.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
        axB1.set_xlim(t_lim)
        axB1.set_xticks(t_ticks)

        axB1.tick_params(axis='y', length=2., pad=2, top=False, labelleft=True, left=True)
        axB1.set_ylim((-1.1,1.1))
        axB1.set_yticks((-1.,0,1))
        axB1.set_ylabel(r"$\mathsf{Re}[\mathcal{E}]~\mathrm{(norm.)}$")

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
        cb = set_colorbar(self.fig, img, axA2, axB2, label='$|\mathcal{E}_\omega|^2\,\mathrm{(dB)}$', dw=0.1)
        cb.set_ticks((-40,-30,-20,-10,0))

        w_ZDW = 2.415
        axA2.axvline( w_ZDW, color='k', dashes=[2,2], linewidth=1)

        axA2.tick_params(axis='x', length=2., pad=2, top=False)

        lam_lim = (0.45,1.35)
        lam_ticks = (0.5,0.7,0.9,1.1,1.3)
        axA2.set_xlim(w_lim)
        axA2.set_xticks(w_ticks)
        axA2.set_xticklabels( (1,2,r'$\omega_{\rm{Z}}$',3,4)  )
        axA2.set_xlabel(r"Angular frequency $\omega\,\mathrm{(rad/fs)}$")

        axA2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
        axA2.set_ylim(z_lim)
        axA2.set_yticks(z_ticks)

        f_name = 'res_spectrum_pyNLO_dz40micron_z14cm.dat'
        dat = np.loadtxt(f_name)
        lam_2 = dat[:,0]/1000
        Iw_2 = dat[:,1]
        w_2 = 2*np.pi*0.29970/lam_2
        l2 = axB2.plot(w_2, _dB(Iw_2/np.max(Iw_2)), color='lightgray', linewidth=1.5, label=r'pyNLO')

        #f_name = 'res_gnlse_Trevors_z14m.dat'
        #dat = np.loadtxt(f_name)
        #w_3 = dat[:,0]
        #Iw_3 = dat[:,1]
        #l3 = axB2.plot(w_3/1000, _dB(Iw_3/np.max(Iw_3)), color='cyan', linewidth=1.5, label=r'gnlse')

        z_id = np.argmin(np.abs(z-14))
        Iw = Iwz_s_f[z_id]
        l1 = axB2.plot(w_s_f, _dB(Iw/np.max(Iw)), color='k', linewidth=0.75, dashes=[2,2], label=r'pyFMAS')

        set_legend(axB2, l1+l2, loc=8, ncol=1)

        w_ZDW = 2.415
        axB2.axvline( w_ZDW, color='k', dashes=[2,2], linewidth=1)

        axB2.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
        axB2.set_xlim(w_lim)
        axB2.set_xticks(w_ticks)

        axB2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False, left=False, right=True, labelright=True)
        axB2.set_ylim((-90,5))
        axB2.set_yticks((-80,-40,0))
        axB2.set_ylabel('$|\mathcal{E}_\omega|^2\,\mathrm{(dB)}$')
        axB2.yaxis.set_label_position('right')

        self.set_subfig_label(axA1,'(d)',loc=1)
        self.set_subfig_label(axA2,'(e)',loc=1)
        self.set_subfig_label(axB1,'(b)',loc=1)
        self.set_subfig_label(axB2,'(c)',loc=1)

    def set_subfig_02(self):
        axD = self.subfig_2

        import fmas.propagation_constant

        beta_fun = fmas.propagation_constant.define_beta_fun_PCF_Ranka2000()
        pc = fmas.propagation_constant.PropConst(beta_fun)

        # -- PROPERTIES OF THE PROPAGATION CONSTANT
        # ... GROUP VELOCITY DISPERSION 

        w_lim = (1.,5)
        w_ticks = (1,2,2.415,3,4,5)
        w = np.linspace(1,5,200)

        b2 = pc.beta2(w)
        l1 = axD.plot(w, b2, color='blue', linewidth=1, label=r'$\beta_2$')

        w_ZDW = 2.415
        axD.axvspan(w_ZDW, w_lim[1], color='lightgray')

        axD.tick_params(axis='x', length=2., pad=2, top=False)
        axD.set_xlim(w_lim)
        axD.set_xticks(w_ticks)
        axD.set_xticklabels( (1,2,r'$\omega_{\rm{Z}}$',3,4,5)  )
        axD.set_xlabel(r"Angular frequency $\omega\,\mathrm{(rad/fs)}$")

        axD.tick_params(axis='y', length=2., pad=2, top=False, labelleft=True, left=True, colors='blue')
        axD.spines['left'].set_color('blue')
        axD.set_ylim(-0.4,0.2)
        axD.set_yticks((-0.4,-0.2,0,0.2))
        axD.set_ylabel(r"GVD $\beta_2\,\mathrm{(fs^2/\mu m)}$",color='blue')

        # ... GROUP DELAY 

        axD2 = axD.twinx()

        b1 = pc.beta1(w)
        l2 = axD2.plot(w, b1, color='red', linewidth=1, dashes=[3,1], label=r'$\beta_1$')

        set_legend(axD, l1,loc=3)
        set_legend(axD2, l2,loc=4)

        axD2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False, labelright=True, left=False, right=True, colors='red')
        axD2.set_ylim(-0.01,0.1)
        axD2.set_yticks((0.,0.04,0.08))
        axD2.set_ylabel(r"rGD $\beta_1\,\mathrm{(fs/\mu m)}$",color='red')

        axD2.spines['left'].set_color('blue')
        axD2.spines['right'].set_color('red')

        self.set_subfig_label(axD,'(a)',loc=1)


    def set_subfig_03(self):
        axC = self.subfig_3

        z_lim = (0,14)
        z_ticks  = (0,4,8,12)

        z,t,w,utz, dz_int, Cp = fetch_data('../res_IFM_SC_Nz4000.h5')
        #z = z*1e-4 # micron -> cm
        dz = z[1]-z[0]
        err = np.abs(Cp[:-1]-Cp[1:])*dz_int/dz/Cp[:-1]
        l1 = axC.plot(z[:-1]*1e-4, err, color='k', linewidth=1., dashes=[3,1,1,1], label=r'$\Delta z = 40\,\mathrm{\mu m}$')

        idx = np.argmax(err)
        print( z[idx], err[idx] )

        z,t,w,utz, dz_int, Cp = fetch_data('../res_IFM_SC_Nz8000.h5')
        #z = z*1e-4 # micron -> cm
        dz = z[1]-z[0]
        err = np.abs(Cp[:-1]-Cp[1:])*dz_int/dz/Cp[:-1]
        l2 = axC.plot(z[:-1]*1e-4, err, color='k', linewidth=1., dashes=[2,1], label=r'$\Delta z = 20\,\mathrm{\mu m}$')

        z,t,w,utz, dz_int, Cp = fetch_data('../res_IFM_SC_Nz16000.h5')
        #z = z*1e-4 # micron -> cm
        dz = z[1]-z[0]
        err = np.abs(Cp[:-1]-Cp[1:])*dz_int/dz/Cp[:-1]
        l3 = axC.plot(z[:-1]*1e-4, err, color='k', linewidth=1., label=r'$\Delta z = 10\,\mathrm{\mu m}$')

        z,t,w,utz, dz_int, Cp = fetch_data('../res_LEM_SC_Nz4000.h5')
        #z = z*1e-4 # micron -> cm
        dz = z[1]-z[0]
        err = np.abs(Cp[:-1]-Cp[1:])*dz_int/dz/Cp[:-1]
        l4 = axC.plot(z[:-1]*1e-4, err, color='red', linewidth=1., linestyle=":", label=r'LEM')

        set_legend(axC,l3+l2+l1+l4,loc=1,ncol=4)

        axC.set_yscale('log')
        axC.set_ylim((1e-14,2*1e-5))
        axC.set_yticks((1e-14,1e-10,1e-6))
        #axC.set_ylim((1e-18,1e-6))
        #axC.set_yticks((1e-18,1e-12,1e-6))
        axC.tick_params(axis='y', length=2., pad=2, top=False)
        axC.set_ylabel(r"Relative error $\delta_{\rm{Ph}}$")

        axC.tick_params(axis='x', length=2., pad=2, top=False)
        axC.set_xlim(z_lim)
        axC.set_xticks((0,2,4,6,8,10,12,14))
        axC.set_xlabel(r"Propagation distance $z\,\mathrm{(cm)}$")

        self.set_subfig_label(axC,'(f)',loc=1)


def fetch_data(file_path):

    with h5py.File(file_path, "r") as f:
        utz = np.array(f['utz'])
        t = np.array(f['t'])
        w = np.array(f['w'])
        z = np.array(f['z'])
        dz_int = np.array(f['dz_integration'])
        Cp = np.array(f['Cp'])
    return z,t,w,utz, dz_int, Cp


def main():

    myFig = Figure(aspect_ratio=1.6, fig_basename='fig_02', fig_format='png')
    myFig.set_layout()
    myFig.set_subfig_01()
    myFig.set_subfig_02()
    myFig.set_subfig_03()
    myFig.save()

if __name__ == '__main__':
    main()
