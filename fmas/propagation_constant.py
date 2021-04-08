"""
The provided module implements a convenience class for performing simple
calculations for a user-defined propagation constant.

.. autosummary::
   :nosignatures:

   PropConst
   define_beta_fun_ZBLAN
   define_beta_fun_ESM
   define_beta_fun_NLPM750
   define_beta_fun_fluoride_glass_AD2010
   define_beta_fun_PCF_Ranka2000
   define_beta_fun_slot_waveguide_Zhang2012


.. module:: propagation_constant

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import scipy
import scipy.optimize as so
import scipy.misc as smi
import scipy.special as ssp
import numpy as np
from .config import C0


class PropConst:
    r"""Convenience class for working with propagation constants.

    Implements methods that provide convenient access to recurrent tasks
    involving propagation constants.

    Args:
        beta_fun (:obj:`callable`):
            Function implementing a propagation constant.

    Attributes:
        beta_fun (:obj:`callable`):
            Function implementing a propagation constant.
        dw (:obj:`int`):
            Angular frequency increment used for calculating derivatives.
        c0 (:obj:`float`):
            Speed of light (default = 0.29970 micron/fs).
    """

    def __init__(self, beta_fun):
        self.dw = 1e-2
        self.c0 = C0
        self.beta_fun = beta_fun

    def beta(self, w):
        """Propagation constant.

        Args:
            w (:obj:`numpy.ndarray`):
                Angular frequency for which to compute propagation constant.

        Returns:
            :obj:`numpy.ndarray` or `float`: Propagation constant.
        """
        return self.beta_fun(w)

    def beta1(self, w):
        """Group delay.

        Args:
            w (:obj:`numpy.ndarray`):
                Angular frequency for which to compute group delay.

        Returns:
            :obj:`numpy.ndarray` or `float`: Group delay.
        """
        return smi.derivative(self.beta_fun, w, dx=self.dw, n=1, order=3)

    def beta2(self, w):
        """Group velocity dispersion (GVD).

        Args:
            w (:obj:`numpy.ndarray`):
                Angular frequency for which to compute GVD.

        Returns:
            :obj:`numpy.ndarray` or `float`: Group velocity dispersion.
        """
        return smi.derivative(self.beta_fun, w, dx=self.dw, n=2, order=5)

    def beta3(self, w):
        """Third order dispersion.

        Args:
            w (:obj:`numpy.ndarray`):
                Angular frequency for which to compute 3rd order dispersion.

        Returns:
            :obj:`numpy.ndarray` or `float`: Group velocity dispersion.
        """
        return smi.derivative(self.beta_fun, w, dx=self.dw, n=3, order=7)

    def vg(self, w):
        r"""Group velocity profile.

        Args:
            w (:obj:`numpy array` or `float`):
                Angular frequency for which to compute group-velocity.

        Returns:
            :obj:`numpy.ndarray` or `float`: Group velocity.
        """
        return 1.0 / self.beta1(w)

    def v_sol_corr(self, t0, w):
        r"""Corrected soliton velocity.

        Computes corrected soliton velocity

        .. math::
            v_{g}^\prime(\omega_0, t_0) = \left[
                \beta_1(\omega_0) -
                \beta_2(\omega_0)/(\omega_0 t_0^2) +
                \beta_3(\omega_0)/(6t_0^2)
                \right]^{-1},

        defined in [PBA2016]_.

        Args:
            t0 (:obj:`float`):
                Soliton duration.
            w (:obj:`numpy array` or `float`):
                Angular frequency for which to compute group-velocity.

        Returns:
            :obj:`numpy.ndarray` or `float`: corrected soliton velocity.

        .. [PBA2016] S. Pickartz, U. Bandelow, Sh. Amiranashvili, Adiabatic theory
            of solitons fed by dispersive waves, Phys. Rev. A 94 (2016) 033811,
            https://doi.org/10.1103/PhysRevA.94.033811.
        """
        b1, b2, b3 = self.beta1(w), self.beta2(w), self.beta3(w)
        return 1.0 / (b1 - b2 / (w * t0 * t0) + b3 / (6 * t0 * t0))

    def lam(self, w):
        r"""Angular frequency to wavelength conversion.

        Args:
            w (:obj:`numpy array` or `float`):
                Angular frequency for which to compute wavelengths.

        Returns:
            :obj:`numpy.ndarray` or `float`: Wavelength values.
        """
        return 2.0 * np.pi * self.c0 / w

    def CD(self, w):
        """Chromatic dispersion.

        Args:
            w (:obj:`numpy array` or `float`):
                Angular frequency for which to compute chromatic dispersion.

        Returns:
            :obj:`numpy.ndarray` or `float`: Chromatic dispersion.
        """
        return -2.0 * np.pi * self.c0 * self.beta2(w) / self.lamb(w) ** 2

    def find_root_beta2(self, w_min, w_max):
        r"""Determine bracketed root of 2nd order dispersion profile.

        Attempts to find a root of the 2nd order dispersion profile in the
        interval from :math:`\omega_{\mathrm{min}}` to
        :math:`\omega_{\mathrm{max}}`.

        Note:
            * Helper method for analysis of dispersion profile
            * Uses scipy.optimize.bisect for bracketed root finding

        Args:
            w_min (:obj:`float`): lower bound for root finding procedure
            w_max (:obj:`float`): upper bound for root finding procedure

        Returns:
            :obj:`float`: root of 2nd order dispersion profile in bracketed
            interval

        """
        return so.bisect(self.beta2, w_min, w_max)

    def find_match_beta1(self, w0, w_min, w_max):
        r"""Determine group velocity matched partner frequency.

        Attempts to find a group-velocity matched partner frequency for
        :math:`\omega_0` in the interval from :math:`\omega_{\mathrm{min}}` to
        :math:`\omega_{\mathrm{max}}`.

        Note:
            * Helper method for analysis of dispersion profile
            * Uses scipy.optimize.minimize_scalar for bracketed minimization
            * If no group velocity matched frequency is contained in the
              supplied interval, the output should not be trusted. Check the
              resulting frequency by assessing whether `beta1(res)==beta1(w0)`.

        Args:
            w0 (:obj:`float`):
                Frequency for which group velocity matched partner frequency
                will be computed.
            w_min (:obj:`float`):
                Lower bound for root finding procedure
            w_max (:obj:`float`):
                Upper bound for root finding procedure

        Returns:
            :obj:`float`: Group-velocity matched partner frequency of `w0`.
        """
        return so.minimize_scalar(
            lambda w: np.abs(self.beta1(w) - self.beta1(w0)),
            bounds=(w_min, w_max),
            method="bounded",
        ).x

    def compute_expansion_coefficients(self, w0, n_max=5):
        r"""Obtain Taylor expansion coefficients at given frequency.

        Note:
            Uses scipy.misc.derivative and scipy.misc.factorial for computing
            the expansion coefficients.

        Args:
            w0 (:obj:`float`):
                    Frequency value about which to expand.
            n_max (:obj:`int`):
                    Number of expansion coefficients to compute (default = 5).

        Returns:
            :obj:`numpy.ndarray`: Taylor expansion coefficients.

        """
        return np.asarray(
            [
                smi.derivative(self.beta, w0, dx=self.dw, n=n, order=2 * n + 1)
                / ssp.factorial(n)
                for n in range(n_max + 1)
            ]
        )

    local_coeffs = compute_expansion_coefficients


def define_beta_fun_ZBLAN():
    r"""Propagation constant for ZBLAN fiber.

    Enclosing function returning a closure implementing a rational
    Pade-approximant of order [4/4] for the refractive index of a ZBLAN fiber
    (PCF).

    Returns:
        :obj:`callable`: Propagation constant for NL-PM-750 PCF.
    """
    p = np.poly1d((11.3882, 0.0, 760.771, 0.0, -1.0)[::-1])
    q = np.poly1d((8.69689, 0.0, 351.039, 0.0, -1.0)[::-1])
    n_idx = lambda w: np.sqrt(p(w) / q(w))  # (-)
    b2Roots = (0.25816960569391839, 1.1756233558942193)
    c0 = 0.29979  # (micron/fs)
    return lambda w: n_idx(w) * w / c0  # (1/micron)


def define_beta_fun_ESM():
    r"""Propagation constant for an ESM PCF.

    Enclosing function returning a closure implementing a rational
    Pade-approximant of order [8/8] for the refractive index of a endlessly
    single mode (ESM) nonlinear photonic crystal fiber (PCF), see [SK2007]_.

    Returns:
        :obj:`callable`: Propagation constant for ESM PCF.

    .. [SK2007] J.M. Stone, J.C. Knight, Visibly 'white' light generation in
       uniform photonic crystal fiber using a microchip laser ,  Optics Express
       16 (2007) 2670.
    """
    p = np.poly1d(
        (16.89475, 0.0, -319.13216, 0.0, 34.82210, 0.0, -0.992495, 0.0, 0.0010671)[::-1]
    )
    q = np.poly1d(
        (1.00000, 0.0, -702.70157, 0.0, 78.28249, 0.0, -2.337086, 0.0, 0.0062267)[::-1]
    )
    n_idx = lambda w: 1 + p(w) / q(w)  # (-)
    c0 = 0.29979  # (micron/fs)
    return lambda w: n_idx(w) * w / c0  # (1/micron)


def define_beta_fun_NLPM750():
    r"""Propagation constant for NLPM750 PCF.

    Enclosing function returning a closure implementing a rational
    Pade-approximant of order [4/4] for the refractive index of a NL-PM-750
    nonlinear photonic crystal fiber (PCF), see [NLPM750]_.


    Returns:
        :obj:`callable`: Propagation constant for NL-PM-750 PCF.


    .. [NLPM750] NL-PM-750 Nonlinear Photonic Crystal Fiber,
       www.nktphotonics.com.
    """
    p = np.poly1d((1.49902, -2.48088, 2.41969, 0.530198, -0.0346925)[::-1])
    q = np.poly1d((1.00000, -1.56995, 1.59604, 0.381012, -0.0270357)[::-1])
    n_idx = lambda w: p(w) / q(w)  # (-)
    c0 = 0.29979  # (micron/fs)
    return lambda w: n_idx(w) * w / c0  # (1/micron)


def define_beta_fun_fluoride_glass_AD2010():
    """Helper function for propagation constant.

    Enclosing function returning a closure implementing a rational
    Pade-approximant of order [5/5] for the refractive index of a fluorid glass
    fiber given in [AD2010]_.

    Returns:
        :obj:`callable`: Propagation constant for NL-PM-750 PCF.

    .. [AD2010] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
      propagation equations for ultrashort optical pulses,  Phys. Rev. A, 82
      (2010) 013812
    """
    p = np.poly1d((1.00654, -2.31431, 1.95942, -0.678111, 0.120882, -0.00911063)[::-1])
    q = np.poly1d((1.00000, -2.29967, 1.94727, -0.673382, 0.120015, -0.00905104)[::-1])
    n_idx = lambda w: p(w) / q(w)  # (-)
    c0 = 0.29979  # (micron/fs)
    return lambda w: n_idx(w) * w / c0  # (1/micron)


def define_beta_fun_PCF_Ranka2000():
    """Helper function for propagation constant.

    Enclosing function returning a closure implementing a polynomial expansion
    of order 11 for the propagation constant of [RWS2000]_ given in [DGC2006]_.

    Returns:
        :obj:`callable`: Propagation constant.

    .. [RWS2000] J. K. Ranka, R. S. Windeler, A. J. Stentz, Visible continuum
        generation in air-silica microstructure optical fibers with anomalous
        dispersion at 800 nm, Opt. Lett. 25 (2000) 25,
        https://doi.org/10.1364/OL.25.000025.

    .. [DGC2006] J. M. Dudley, G. Genty, S. Coen, Supercontinuum generation in
        photonic crystal fiber, Rev. Mod. Phys. 78 (2006) 1135,
        http://dx.doi.org/10.1103/RevModPhys.78.1135
    """
    # EXPANSION COEFFICIENTS DISPERSION
    b2 = -1.1830e-2  # (fs^2/micron)
    b3 = 8.1038e-2  # (fs^3/micron)
    b4 = -0.95205e-1  # (fs^4/micron)
    b5 = 2.0737e-1  # (fs^5/micron)
    b6 = -5.3943e-1  # (fs^6/micron)
    b7 = 1.3486  # (fs^7/micron)
    b8 = -2.5495  # (fs^8/micron)
    b9 = 3.0524  # (fs^9/micron)
    b10 = -1.7140  # (fs^10/micron)
    # PROPAGATION CONSTANT (DEPENDING ON DETUNING)
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
    # REFERENCE FREQUENCY FOR WHICH EXPANSION IS VALID
    w_ref = 2.2559  # (rad/fs)
    return lambda w: beta_fun_detuning(w - w_ref)


def define_beta_fun_slot_waveguide_Zhang2012():
    """Helper function for propagation constant.

    Enclosing function returning a closure implementing a polynomial expansion
    of order 8 for the propagation constant of [Z2012]_.

    Returns:
        :obj:`callable`: Propagation constant.

    .. [Z2012] Zhang et al., Silicon waveguide with four zero-dispersion
        wavelengths and its application in on-chip octave spanning
        supercontinuum generation, Opt. Express 20 (2012) 1685.
    """
    beta_fun = np.poly1d(
        [
            39.75719715,
            -368.37371075,
            1489.51296969,
            -3431.76080305,
            4925.79793757,
            -4508.99480626,
            2569.75813512,
            -819.6299219,
            0.0,
        ]
    )
    return beta_fun
