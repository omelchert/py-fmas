"""
Implements a custom model for a specific photonic crystal fiber.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .fmas_s_raman import FMAS_S_Raman
from ..config import FTFREQ, FT, IFT, C0


class CustomModelPCF(FMAS_S_Raman):
    r"""Custom model for specific Photonic Crystal Fiber"""

    def __init__(self, w):

        c0 = 0.29979  # (micron/fs)
        w0 = 2.2559  # (rad/fs)
        gam0 = 0.11e-6  # (1/W/micron)
        fR = 0.18  # (-)
        tau1 = 12.2  # (fs)
        tau2 = 32.0  # (fs)
        n2 = gam0 * c0 / w0

        def _beta_fun_detuning(w):
            r"""Helper function for propagation constant

            Implements group-velocity dispersion with expansion coefficients
            listed in Tab. I of Ref. [1]. Expansion coefficients are valid for
            :math:`lambda = 835\,\mathrm{nm}`, i.e. for :math:`\omega_0 \approx
            2.56\,\mathrm{rad/fs}`.

            References:
                [1] J. M. Dudley, G. Genty, S. Coen,
                Supercontinuum generation in photonic crystal fiber,
                Rev. Mod. Phys. 78 (2006) 1135,
                http://dx.doi.org/10.1103/RevModPhys.78.1135

            Args:
                w (:obj:`numpy.ndarray`): Angular frequency grid.

            Returns:
                :obj:`numpy.ndarray` Propagation constant as function of
                frequency detuning.
            """
            # ... EXPANSION COEFFICIENTS DISPERSION
            b2 = -1.1830e-2  # (fs^2/micron)
            b3 = 8.1038e-2  # (fs^3/micron)
            b4 = -0.95205e-1  # (fs^4/micron)
            b5 = 2.0737e-1  # (fs^5/micron)
            b6 = -5.3943e-1  # (fs^6/micron)
            b7 = 1.3486  # (fs^7/micron)
            b8 = -2.5495  # (fs^8/micron)
            b9 = 3.0524  # (fs^9/micron)
            b10 = -1.7140  # (fs^10/micron)
            # ... PROPAGATION CONSTANT (DEPENDING ON DETUNING)
            beta_fun_detuning = np.poly1d(
                [
                    b10 / 3628800,
                    b9 / 362880,
                    b8 / 40320,
                    b7 / 5040,
                    b6 / 720,
                    b5 / 120,
                    b4 / 24,
                    b3 / 6,
                    b2 / 2,
                    0.0,
                    0.0,
                ]
            )
            return beta_fun_detuning(w)

        beta_w = _beta_fun_detuning(w - w0)
        # -- EQUIP THE SUPERCLASS
        super().__init__(w, beta_w, n2, fR, tau1, tau2)
