"""
This module implements functions for postprocessing of simulation data.

.. autosummary::
   :nosignatures:

    change_reference_frame
    spectrogram
    plot_spectrogram
    plot_evolution
    plot_details_prop_const

.. module:: tools

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import sys
import time
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from .config import FT, IFT, FTFREQ, FTSHIFT

# -- CUSTOM SECH-FUNCTION
# prevents "RuntimeWarning: overflow encountered in cosh" by setting
# the evaluation of 1./cosh(x) to zero where appropriate, i.e. if cosh
# evaluates to a float that is larger than the largest representable
# float fMax = numpy.finfo(float).max in numpy. -- 2019-07-07 OM
sech = np.vectorize(lambda x: 1.0 / np.cosh(x) if abs(x) < 710.4 else 0.0)


class ProgressBar:
    def __init__(self, num_iter, bar_len=60):
        self.it_max = num_iter
        self.bar_len = bar_len
        self.t_ini = time.perf_counter()

    def update(self, it):
        t_curr = time.perf_counter()
        perc = 100 * it / self.it_max
        it_per_sec = 1.0 / (t_curr - self.t_ini)
        sec_to_end = (self.it_max - it) / it_per_sec
        sys.stderr.write("\r")
        sys.stderr.write(
            "[{:<{}}] {:.0f}% ({:d}/{:d}) it; {:.1f} it/s; {:.1f} s".format(
                "=" * int(self.bar_len * perc / 100),
                self.bar_len,
                perc,
                it,
                self.it_max,
                it_per_sec,
                sec_to_end,
            )
        )
        sys.stderr.flush()
        self.t_ini = t_curr

    def finish(self):
        sys.stderr.write("\n")


def change_reference_frame(w, z, uwz, v0):
    r"""Change reference frame.

    Shift to moving frame of reference in which the dynamics is slow.

    Args:
        w (:obj:`numpy.ndarray`): Angular-frequency grid.
        z (:obj:`numpy.ndarray`): :math:`z`-grid.
        uwz (:obj:`numpy.ndarray`, 2-dim): Frequency domain representation of
            analytic signal.
        v0 (:obj:`float`): Reference velocity.
    """
    return IFT(uwz * np.exp(-1j * w * z[:, np.newaxis] / v0), axis=-1)


def spectrogram(t, w, ut, t_lim=None, Nt=1000, Nw=2 ** 8, s0=20.0):
    """Compute spectrogram for time-domain input signal.

    Computes spectrogram of a time-domain input signal via short time Fourier
    transform employing a Gaussian window function.

    Args:
        t (:obj:`numpy.array`, 1-dim):
              Temporal grid.
        w (:obj:`numpy.array`, 1-dim):
              Angular-frequency grid.
        Et (:obj:`numpy-array`, 1-dim):
              Time-domain representation of analytic signal.
        t_lim (:obj:`list`):
              Delay time bounds for temporal axis considered for constructing
              the spectrogram (tMin, tMax), default is (min(t),max(t)).
        Nt (:obj:`int`):
              Number of delay times samples in [tMin, tMax], used for signal
              localization (default: Nt=1000).
        Nw (:obj:`int`):
              Number of samples in angular-frequency domain kept as output
              (default: Nw=256).
        s0 (:obj:`float`):
              Root-mean-square width of Gaussian function used for signal
              localization (default: s0=20.0).

    Returns:
        :obj:`list`: (t_spec, w_spec, P_tw), where `t_seq`
        (:obj:`numpy.ndarray`, 1-dim) are delay times, `w`
        (:obj:`numpy.ndarray`, 1-dim) are angular frequencies, and `P_tw`
        (:obj:`numpy.ndarray`, 2-dim) is the spectrogram.
    """
    if t_lim == None:
        t_min, t_max = np.min(t), np.max(t)
    else:
        t_min, t_max = t_lim
    # -- DELAY TIMES
    t_seq = np.linspace(t_min, t_max, Nt)
    # -- WINDOW FUNCTION
    h = lambda t: np.exp(-(t ** 2) / 2 / s0 / s0) / np.sqrt(2.0 * np.pi * s0 * s0)
    # -- COMPUTE TIME-FREQUENCY RESOLVED CONTENT OF INPUT FIELD
    P = np.abs(FT(h(t - t_seq[:, np.newaxis]) * ut[np.newaxis, :], axis=-1)) ** 2
    return t_seq, FTSHIFT(w), np.swapaxes(FTSHIFT(P, axes=-1), 0, 1)


def plot_spectrogram(t_delay, w_opt, P_tw):
    r"""Generate a figure of a spectrogram.

    Generate figure showing the intensity normalized spectrogram.  Scales the
    spectrogram data so that maximum intensity per time and frequency is unity.

    Args:
        t_delay (:obj:`numpy.ndarray`, 1-dim): Delay time grid.
        w_opt (:obj:`numpy.ndarray`, 1-dim): Angular-frequency grid.
        P_tw (:obj:`numpy.ndarray`, 2-dim): Spectrogram data.
    """
    t_min, t_max = t_delay[0], t_delay[-1]
    w_min, w_max = w_opt[0], w_opt[-1]

    f, ax1 = plt.subplots(1, 1, sharey=True, figsize=(4, 3))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.78)
    cmap = mpl.cm.get_cmap("jet")

    def _setColorbar(im, refPos):
        """colorbar helper"""
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = f.add_axes([x0, y0 + 1.02 * h, w, 0.05 * h])
        cbar = f.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(
            color="k",
            labelcolor="k",
            bottom=False,
            direction="out",
            labelbottom=False,
            labeltop=True,
            top=True,
            size=4,
            pad=0,
        )

        cbar.ax.tick_params(which="minor", bottom=False, top=False)
        return cbar

    I = P_tw[:-1, :-1] / P_tw.max()
    im1 = ax1.pcolorfast(
        t_delay,
        w_opt,
        I,
        norm=col.LogNorm(vmin=1e-5 * I.max(), vmax=I.max()),
        cmap=cmap,
    )
    cbar1 = _setColorbar(im1, ax1.get_position())
    cbar1.ax.set_title(r"$P_S(t, \omega)$", color="k", y=3.5)

    ax1.set_xlim(t_min, t_max)
    ax1.set_ylim(w_min, w_max)
    ax1.tick_params(axis="y", length=2.0, direction="out")
    ax1.tick_params(axis="x", length=2.0, direction="out")
    ax1.set_xlabel(r"Delay time $t$")
    ax1.set_ylabel(r"Angular frequency $\omega$")

    plt.show()


def plot_evolution(z, t, u, t_lim=None, w_lim=None, DO_T_LOG=False, ratio_Iw=1e-6):
    """Generate a figure of a pulse propagation scenario.

    Generates a plot showing the z-propagation characteristics of
    the squared magnitude field envelope (left subfigure) and
    the spectral intensity (right subfigure).

    Args:
        z (:obj:`numpy.ndarray`):
            :math:`z`-grid.
        t (:obj:`numpy.ndarray`):
            Temporal grid.
        u (:obj:`numpy.ndarray`):
            Time-domain representation of analytil signal.
        t_lim (:obj:`list`, 2-tuple):
            Time range in the form (t_min, t_max) (default=None).
        w_lim (:obj:`list`, 2-tuple):
            Angular frequency range in the form (w_min,w_max) (default=None).
        DO_T_LOG (:obj:`bool`):
            Flag indicating whether time-domain propagation characteristics
            will be shown on log-scale (default=True).
    """

    def _setColorbar(im, refPos):
        """colorbar helper"""
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = f.add_axes([x0, y0 + 1.02 * h, w, 0.03 * h])
        cbar = f.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(
            color="k",
            labelcolor="k",
            bottom=False,
            direction="out",
            labelbottom=False,
            labeltop=True,
            top=True,
            size=4,
            pad=0,
        )

        cbar.ax.tick_params(which="minor", bottom=False, top=False)
        return cbar

    def _truncate(I):
        """truncate intensity

        fixes python3 matplotlib issue with representing small
        intensities on plots with log-colorscale
        """
        I[I < 1e-20] = 1e-20
        return I

    w = nfft.ifftshift(nfft.fftfreq(t.size, d=t[1] - t[0]) * 2 * np.pi)

    if t_lim == None:
        t_lim = (np.min(t), np.max(t))
    if w_lim == None:
        w_lim = (np.min(w), np.max(w))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    plt.subplots_adjust(left=0.13, right=0.96, bottom=0.12, top=0.8, wspace=0.05)
    cmap = mpl.cm.get_cmap("jet")

    # -- LEFT SUB-FIGURE: TIME-DOMAIN PROPAGATION CHARACTERISTICS
    It = np.abs(u) ** 2
    It /= np.max(It)
    It = _truncate(It)

    if DO_T_LOG:
        my_norm = col.LogNorm(vmin=1e-6 * It.max(), vmax=It.max())
    else:
        my_norm = col.Normalize(vmin=0, vmax=1)

    im1 = ax1.pcolorfast(t, z, It[:-1, :-1], norm=my_norm, cmap=cmap)
    cbar1 = _setColorbar(im1, ax1.get_position())
    cbar1.ax.set_title(r"$|u|^2/{\rm{max}}\left(|u|^2\right)$", color="k", y=3.5)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.set_xlim(t_lim)
    ax1.set_ylim([0.0, z.max()])
    ax1.set_xlabel(r"Time $t$")
    ax1.set_ylabel(r"Propagation distance $z$")
    ax1.ticklabel_format(useOffset=False, style="plain")

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS
    Iw = np.abs(nfft.ifftshift(FT(u, axis=-1), axes=-1)) ** 2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax2.pcolorfast(
        w,
        z,
        Iw[:-1, :-1],
        norm=col.LogNorm(vmin=ratio_Iw * Iw.max(), vmax=Iw.max()),
        cmap=cmap,
    )
    cbar2 = _setColorbar(im2, ax2.get_position())
    cbar2.ax.set_title(
        r"$|u_\omega|^2/{\rm{max}}\left(|u_\omega|^2\right)$", color="k", y=3.5
    )
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")
    ax2.set_xlim(w_lim)
    ax2.set_ylim([0.0, z.max()])
    ax2.set_xlabel(r"Angular frequency $\omega$")
    ax2.tick_params(labelleft=False)
    ax2.ticklabel_format(useOffset=False, style="plain")

    plt.show()


def plot_details_prop_const(w, vg, beta2):
    """Generate a figure of the group-velocity and group-velocity dispersion.

    Generates a plot showing the grop-velocity (top subplot) and group-velocity
    dispersion (bottom subplot).

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        vg (:obj:`numpy.ndarray`):
            Group-velocity profile.
        beta2 (:obj:`numpy.ndarray`):
            Group-velocity dispersion profile.
    """

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 4))
    plt.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.96, hspace=0.1)

    l1 = ax1.plot(w, vg, color="k", linewidth=1)

    ax1.set_xlim(np.min(w), np.max(w))
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.tick_params(axis="y", length=2.0)
    ax1.tick_params(axis="x", length=2.0, labelbottom=False)
    ax1.set_ylabel(r"GV $v_g~\mathrm{(\mu m/fs)}$")

    l2 = ax2.plot(w, beta2, color="k", linewidth=1)
    ax2.axhline(0, color="k", lw=0.75, ls=":")

    ax2.set_xlim(np.min(w), np.max(w))
    ax2.ticklabel_format(useOffset=False, style="plain")
    ax2.tick_params(axis="y", length=2.0)
    ax2.tick_params(axis="x", length=2.0)
    ax2.set_ylabel(r"GVD $\beta_2~\mathrm{(fs^2/\mu m)}$")
    ax2.set_xlabel(r"Angular frequency $\omega~\mathrm{(rad/fs)}$")

    plt.show()
