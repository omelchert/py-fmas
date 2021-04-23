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

__author__ = "Oliver Melchert"
__date__ = "2020-09-09"


# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft


class Figure:
    def __init__(self, aspect_ratio=1.0, fig_format="png", fig_basename="fig_test"):
        self.fig_format = fig_format
        self.fig_basename = fig_basename
        self.fig = None
        self.set_style(aspect_ratio)

    def set_style(self, aspect_ratio=1.0):

        fig_width = 3.4  # (inch)
        fig_height = aspect_ratio * fig_width

        params = {
            "figure.figsize": (fig_width, fig_height),
            "legend.fontsize": 6,
            "legend.frameon": False,
            "axes.labelsize": 7,
            "axes.linewidth": 1.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "mathtext.fontset": "stixsans",
            "mathtext.rm": "serif",
            "mathtext.bf": "serif:bold",
            "mathtext.it": "serif:italic",
            "mathtext.sf": "sans\\-serif",
            "font.size": 7,
            "font.family": "serif",
            "font.serif": "Helvetica",
        }
        mpl.rcParams.update(params)

    def save(self):
        fig_format, fig_name = self.fig_format, self.fig_basename
        if fig_format == "png":
            plt.savefig(fig_name + ".png", format="png", dpi=600)
        elif fig_format == "pdf":
            plt.savefig(fig_name + ".pdf", format="pdf", dpi=600)
        elif fig_format == "svg":
            plt.savefig(fig_name + ".svg", format="svg", dpi=600)
        else:
            plt.show()

    def set_subfig_label(self, ax, label, loc=1):
        pos = ax.get_position()

        if loc == 1:
            self.fig.text(
                pos.x0,
                pos.y1,
                label,
                color="white",
                backgroundcolor="k",
                bbox=dict(facecolor="k", edgecolor="none", boxstyle="square,pad=0.1"),
                verticalalignment="top",
            )

        elif loc == 2:
            self.fig.text(
                pos.x0,
                pos.y0,
                label,
                color="white",
                backgroundcolor="k",
                bbox=dict(facecolor="k", edgecolor="none", boxstyle="square,pad=0.1"),
                verticalalignment="bottom",
            )

        else:
            print("check label position")
            exit()

    def set_layout(self):

        fig = plt.figure()
        self.fig = fig

        plt.subplots_adjust(
            left=0.12,  # pos. of subplots left border
            bottom=0.15,  # pos. of subplots bottom border
            right=0.88,  # pos. of subplots right border
            top=0.9,  # pos. of subplots top border
            wspace=0.6,  # horizontal space between supblots
            hspace=0.6,  # vertical space between subplots
        )

        gs00 = GridSpec(nrows=1, ncols=1)

        gsA = GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs00[0, 0], wspace=0.1, hspace=0.1
        )
        axA1 = fig.add_subplot(gsA[:, 0])
        axA2 = fig.add_subplot(gsA[0, 1:])
        axA3 = fig.add_subplot(gsA[1, 1:])
        self.subfig_1 = [axA1]
        self.subfig_2 = [axA2, axA3]

    def set_subfig_1(self):
        axA1 = self.subfig_1[0]
        fig = self.fig

        # CONVENIENT FUNCTIONS ##################################################################

        def _set_colorbar(fig, img, ax, label="text", dw=0.0):
            # -- extract position information for colorbar placement
            refPos = ax.get_position()
            x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
            # -- set new axes as reference for colorbar
            colorbar_axis = fig.add_axes([x0 + dw, y0 + 1.02 * h, w - dw, 0.03 * h])
            # -- set custom colorbar
            colorbar = fig.colorbar(
                img,  # image described by colorbar
                cax=colorbar_axis,  # reference axex
                orientation="horizontal",  # colorbar orientation
                extend="both",  # ends with out-of range values
            )
            colorbar.ax.tick_params(
                color="k",  # tick color
                labelcolor="k",  # label color
                bottom=False,  # no ticks at bottom
                labelbottom=False,  # no labels at bottom
                labeltop=True,  # labels on top
                top=True,  # ticks on top
                direction="out",  # place ticks outside
                length=2,  # tick length in pts.
                labelsize=5.0,  # tick font in pts.
                pad=1.0,  # tick-to-label distance in pts.
            )
            fig.text(
                x0,
                y0 + h + 0.02 * h,
                label,
                horizontalalignment="left",
                verticalalignment="bottom",
                size=6,
            )
            return colorbar

        _norm = lambda x: x / x[0].max()
        _truncate = lambda x: np.where(x > 1.0e-20, x, 1.0e-20)
        _dB = lambda x: np.where(x > 1e-20, 10.0 * np.log10(x), 10 * np.log10(1e-20))

        # -- CREATE CUSTOM COLORMAP
        my_cmap = mpl.cm.get_cmap("jet")

        # -- data -------------------------------------------------------------------------------
        z, t, w, utz = self.z, self.t, self.w, self.utz

        # -- time domain
        Itz = _norm(np.abs(utz) ** 2)
        img = axA1.pcolorfast(
            t, z, _dB(Itz[:-1, :-1]), vmin=-40.0, vmax=1, cmap=my_cmap
        )
        cb = _set_colorbar(fig, img, axA1, label="$|A|^2~\mathrm{(dB)}$", dw=0.1)
        cb.set_ticks((-40, -20, 0))

        axA1.tick_params(axis="x", length=2.0, pad=2, top=False)
        x_lim = (-30.0, 30.0)
        x_ticks = (-20.0, 0, 20.0)
        axA1.set_xlim(x_lim)
        axA1.set_xticks(x_ticks)
        axA1.set_xlabel(r"Time $t~\mathrm{(fs)}$")

        axA1.tick_params(axis="y", length=2.0, pad=2, top=False)
        z_lim = (0.0, z.max())
        z_ticks = (0, 0.5, 1.0, 1.5)
        axA1.set_ylim(z_lim)
        axA1.set_yticks(z_ticks)
        axA1.set_ylabel(r"Propagation distance $z~\mathrm{(\mu m)}$")

        self.set_subfig_label(axA1, "(a)")

    def set_subfig_2(self):
        axA2, axA3 = self.subfig_2

        z, dz_a, del_rle = self.z, self.dz_a, self.del_rle

        iters = np.arange(dz_a.size)
        axA2.plot(iters, dz_a * 1e3, color="k", lw=0.75)

        axA2.tick_params(axis="x", length=2.0, pad=2, top=False, labelbottom=False)
        x_lim = (0, 512)
        x_ticks = (0, 128, 256, 384, 512)
        axA2.set_xlim(x_lim)
        axA2.set_xticks(x_ticks)
        axA2.set_ylabel(r"$h~{(\mathrm{ns})}$")
        axA2.yaxis.set_label_position("right")

        axA2.tick_params(
            axis="y",
            length=2.0,
            pad=2,
            top=False,
            labelleft=False,
            right=True,
            labelright=True,
            left=False,
        )
        y_lim = (0, 8.5)
        y_ticks = (0, 2, 4, 6, 8)
        axA2.set_ylim(y_lim)
        axA2.set_yticks(y_ticks)

        axA3.plot(iters, del_rle * 1e7, color="k", lw=0.75)
        axA3.axhspan(0.5, 1, color="lightgray")

        axA3.tick_params(axis="x", length=2.0, pad=2, top=False)
        y_lim = (0, 1.7)
        y_ticks = (0, 0.5, 1, 1.5)
        axA3.set_xlim(x_lim)
        axA3.set_xticks(x_ticks)

        axA3.tick_params(
            axis="y",
            length=2.0,
            pad=2,
            top=False,
            labelleft=False,
            right=True,
            labelright=True,
            left=False,
        )
        axA3.set_ylim(y_lim)
        axA3.set_yticks(y_ticks)
        axA3.set_ylabel(r"$\delta_{\rm{RLE}}~(\times 10^{-7})$")
        axA3.yaxis.set_label_position("right")
        axA3.set_xlabel(r"Slice number $n$")

        self.set_subfig_label(axA2, "(b)")
        self.set_subfig_label(axA3, "(c)")

    def fetch_data(self, file_path):

        with h5py.File(file_path, "r") as f:
            self.utz = np.array(f["u"])
            self.t = np.array(f["t"])
            self.w = np.array(f["w"])
            self.z = np.array(f["z"])
            self.dz_int = np.array(f["dz_integration"])
            self.dz_a = np.array(f["dz_a"])
            self.del_rle = np.array(f["del_rle"])


def main():

    myFig = Figure(
        aspect_ratio=0.66, fig_basename="fig_LEM_SolSolCollision", fig_format="png"
    )
    myFig.set_layout()
    myFig.fetch_data("../res_LEM_SolSolCollision.h5")
    myFig.set_subfig_1()
    myFig.set_subfig_2()
    myFig.save()


if __name__ == "__main__":
    main()
