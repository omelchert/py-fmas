"""
Module containing helper routines, convenient abbreviations, and constants.

.. module:: config

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
import numpy.fft as nfft

# -- FAST FOURIER-TRANSFORM
FT = nfft.ifft
r"""Compute one-dimensional discrete Fourier Transform (DFT).

Note:
    * Alias for `numpy.fft.ifft`.
    * See numpy.ifft for definition of DFT, its arguments and conventions.
    * See ifft for inverse of `fft`.
"""

IFT = nfft.fft
r"""Compute one-dimensional inverse discrete Fourier Transform (DFT).

Note:
    * Alias for `numpy.fft.fft`.
    * See numpy.fft for definition of DFT, its arguments and conventions.
    * See FT for inverse of IFT.
"""

FTFREQ = nfft.fftfreq
r"""Discrete Fourier Transform sample frequencies.

Note:
    * Alias for `numpy.fft.fftfreq`.
    * See numpy.fft.fftfreq for definition, arguments and conventions.
"""

FTSHIFT = nfft.fftshift
r"""Shift order of frequencies.

Note:
    * Alias for `numpy.fft.fftshift`.
    * See numpy.fft.fftshift for definition, arguments and conventions.
"""

# -- ANGULAR FREQUENCY BOUND FOR DEALIASING
W_MAX_FAC = 0.75
r"""float: Angular frequency bound for dealiasing.

All angular frequency components satisfying abs(w) >= W_MAX_FAC*max(w) are
discarded.
"""

# -- C0 - UNIT(C0) = micron/fs = mm/ps
C0 = 0.29979
r"""float: Speed of light.

Units are [C0] = micron/fs.
"""
