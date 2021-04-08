"""
This module implements several Raman response functions.  Numerical models of
the Raman response are important for the accurate theoretical description of
the propagation of optical pulses with short duration and high peak power
[MM1986]_ [G1986]_.

The following Raman response models are currently supported:

.. autosummary::
   :nosignatures:

    h_BW
    h_LA
    h_HC

.. [MM1986] F. M. Mitschke, L. F. Mollenauer, Discovery of the soliton
    self-frequency shift, Opt. Lett. 11 (1986) 659,
    https://doi.org/10.1364/OL.11.000659.

.. [G1986] J. P. Gordon, Theory of the soliton self-frequency shift, Opt. Lett. 11
    (1986) 662, https://doi.org/10.1364/OL.11.000662.

.. [BW1989] K. J. Blow, D. Wood, Theoretical description of transient
    stimulated Raman scattering in optical fibers.  IEEE J. Quantum Electron.,
    25 (1989) 1159, https://doi.org/10.1109/3.40655.

.. [LA2006] Q. Lin, G. P. Agrawal, Raman response function for silica fibers,
    Optics Letters, 31 (2006) 3086, https://doi.org/10.1364/JOSAB.6.001159.

.. [HC2002] D. Hollenbeck and C. D. Cantrell, Multiple-vibrational-mode mopdel
    for fiber-optic Raman gain spectrum and response function, J. Opt.
    Soc.  Am. B, 19 (2002) 2886, https://doi.org/10.1364/JOSAB.19.002886.

.. module:: raman_response

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import sys
import numpy as np
from .config import FTFREQ, FT, IFT


def h_BW(t, tau1=12.2, tau2=32.0):
    r"""Blow-Wood type Raman response function [BW1989]_.

    Implements simple Raman response function for silica fibers based on a
    single damped harmonic oscillator with Lorentzian linewidth [BW1989]_,
    given by

    .. math::
        h_{\mathrm{BW}}(t) = \frac{\tau_1^2 + \tau_2^2}{\tau_1\tau_2^2}\, e^{-t/\tau_2}\, \sin(t/\tau_1)\,\theta(t),

    where causality is assured by the unit step fuction :math:`\theta(t)`.
    This Raman response model can be adapted to fit various types of nonlinear
    fibers. For example, using the parameters :math:`\tau_1=12.2\,\mathrm{fs}`,
    and :math:`\tau_2=32\,\mathrm{fs}`, toghether with a fractional Raman
    contribution :math:`f_R=0.18` is adequate for modeling the Raman response
    of of silica fibers.

    Args:
        t (:obj:`numpy.ndarray`): temporal grid.
        tau1 (:obj:`float`): Raman response parameter (default: 12.2 fs).
        tau2 (:obj:`float`): Raman response parameter (default: 32.0 fs).

    Returns:
        :obj:`numpy.ndarray`: Angular-frequency representation of the Raman response.
    """
    w = FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi
    hR = np.where(
        t > 0,
        (tau1 ** 2 + tau2 ** 2)
        / (tau1 * tau2 ** 2)
        * np.exp(-t / tau2)
        * np.sin(t / tau1),
        0,
    )
    hR /= np.sum(hR)
    return np.exp(1j * w * np.min(t)) * FT(hR) * t.size


def h_LA(t):
    r"""Lin-Agrawal type Raman response function [LA2006]_.

    Implements an improved Raman response model, taking into account the
    anisotropic nature of Raman scattering [LA2006]_, given by

    .. math::
        h_{\mathrm{LA}}(t) = (1-f_b)\,h_{\mathrm{BW}}(t) +f_b\,\frac{2\tau_b-t}{\tau_b^2} e^{-t/\tau_b}\,\theta(t),

    with :math:`h_{\mathrm{BW}}(t)` given by :class:`h_BW`, and parameters
    :math:`\tau_b=96\,\mathrm{fs}`, and :math:`f_b = 0.21`.

    Args:
        t (:obj:`numpy.ndarray`): temporal grid.

    Returns:
        :obj:`numpy.ndarray`: Angular-frequency representation of the Raman response.
    """
    tau1 = 12.2  # (fs)
    tau2 = 32.0  # (fs)
    taub = 96.0  # (fs)
    fb = 0.21
    w = FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi
    ha = tau1 * (tau1 ** (-2) + tau2 ** (-2)) * np.exp(-t / tau2) * np.sin(t / tau1)
    hb = (2.0 * taub - t) / taub / taub * np.exp(-t / taub)
    hR = np.where(t > 0, (1.0 - fb) * ha + fb * hb, 0)
    hR /= np.sum(hR)
    return np.exp(1j * w * np.min(t)) * FT(hR) * t.size


def h_HC(t):
    r"""Hollenbeck-Cantrell type Raman response function [HC2002_].

    Implements intermediate broadening model for Raman response function of
    silica fibers based on multiple vibrational frequency modes of the Si-O-Si
    compound [HC2002]_. The time-domain representation of this Raman
    response model is given by

    .. math::
        h_{\mathrm{HC}}(t) = \sum_{n=1}^{13} A_n\, e^{-\gamma_n t - \Gamma_n^2 t^2/4}\,\sin(\omega_n t)\,\theta(t),

    with parameter sequences :math:`\{\omega_n\}_{n=1}^{13}`,
    :math:`\{A_n\}_{n=1}^{13}`, :math:`\{\gamma_n\}_{n=1}^{13}`, and
    :math:`\{\Gamma_n\}_{n=1}^{13}`, summarized in the table below.

    .. csv-table::
       :header: n, omega_n (rad/fs), A_n (-), gamma_n (1/ps), Gamma_n (1/ps)
       :widths: 10, 30, 30, 30, 30

       1 ,  0.01060,     1.00,       1.64,        4.91
       2 ,  0.01884,    11.40,       3.66,       10.40
       3 ,  0.04356,    36.67,       5.49,       16.48
       4 ,  0.06828,    67.67,       5.10,       15.30
       5 ,  0.08721,    74.00,       4.25,       12.75
       6 ,  0.09362,     4.50,       0.77,        2.31
       7 ,  0.11518,     6.80,       1.30,        3.91
       8 ,  0.13029,     4.60,       4.87,       14.60
       9 ,  0.14950,     4.20,       1.87,        5.60
       10,  0.15728,     4.50,       2.02,        6.06
       11,  0.17518,     2.70,       4.71,       14.13
       12,  0.20343,     3.10,       2.86,        8.57
       13,  0.22886,     3.00,       5.02,       15.07


    Args:
        t (:obj:`numpy.ndarray`): temporal grid.

    Returns:
        :obj:`numpy.ndarray`: Angular-frequency representation of the Raman response.
    """
    # PARAMTERS FOR INTERMEDIATE BROADENING MODEL - TAB. 1, REF. [1]
    # param: (Comp. Pos, Peak Int., Gaussian FWHM, Lorentz FWHM)
    # units: (    cm^-1,      a.u.,         cm^-1,        cm^-1)
    pArr = [
        (56.25, 1.00, 52.10, 17.37),
        (100.00, 11.40, 110.42, 38.81),
        (231.25, 36.67, 175.00, 58.33),
        (362.50, 67.67, 162.50, 54.17),
        (463.00, 74.00, 135.33, 45.11),
        (497.00, 4.50, 24.50, 8.17),
        (611.50, 6.80, 41.50, 13.83),
        (691.67, 4.60, 155.00, 51.67),
        (793.67, 4.20, 59.50, 19.83),
        (835.00, 4.50, 64.30, 21.43),
        (930.00, 2.70, 150.00, 50.00),
        (1080.00, 3.10, 91.00, 30.33),
        (1215.00, 3.00, 160.00, 53.33),
    ]

    pos, A, FWHMGauss, FWHMLorentz = zip(*pArr)

    c0 = 0.000029979  # cm/fs
    wv = 2 * np.pi * c0 * np.asarray(pos)
    Gamma = np.pi * c0 * np.asarray(FWHMGauss)
    gamma = np.pi * c0 * np.asarray(FWHMLorentz)

    hR = np.zeros(t.size)
    for i in range(len(pArr)):
        hR += (
            A[i]
            * np.exp(-gamma[i] * t)
            * np.exp(-Gamma[i] ** 2 * t ** 2 / 4)
            * np.sin(wv[i] * t)
        )

    hR[t < 0] = 0
    hR /= np.sum(hR)
    w = FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi
    return np.exp(1j * w * np.min(t)) * FT(hR) * t.size


# EOF: raman_response.py
