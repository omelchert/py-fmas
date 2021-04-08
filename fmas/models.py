"""
Implements several :math:`z`-propagation models based on the forward model for
the analutic signal [1,2,3,4].

.. autosummary::
   :nosignatures:

   ModelBaseClass
   FMAS_THG
   FMAS
   FMAS_S
   FMAS_S_Raman
   BMCF
   CustomModelPCF

Further :math:`z`-propagation models can be implemented by using the
class :class:`ModelBaseClass`.

References:
    [1] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
    propagation equations for ultrashort optical pulses, Phys. Rev. E 10
    (2010) 013812, http://dx.doi.org/10.1103/PhysRevA.82.013812.

    [2] Sh. Amiranashvili, A. Demircan, Ultrashort Optical Pulse Propagation in
    terms of Analytic Signal, Adv. Opt. Tech. 2011 (2011) 989515,
    http://dx.doi.org/10.1155/2011/989515.

    [3] A. Demircan, Sh. Amiranashvili, C. Bree, C. Mahnke, F. Mitschke, G.
    Steinmeyer, Rogue wave formation by accelerated solitons at an optical
    event horizon, Appl. Phys. B 115 (2014) 343,
    http://dx.doi.org/10.1007/s00340-013-5609-9

    [4] A. Demircan, Sh. Amiranashvili, C. Bree, U. Morgner, G. Steinmeyer,
    Supercontinuum generation by multiple scatterings at a group velocity
    horizon, Opt. Exp. 22 (2014) 3866,
    https://doi.org/10.1364/OE.22.003866.

.. module:: models

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .config import FTFREQ, FT, IFT, C0


class ModelBaseClass:
    r"""Base class for propagation models.

    Attributes:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Frequency-domain representation of propagation constant.
        c0 (:obj:`float`): speed of light

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Frequency-domain representation of propagation constant.
    """

    def __init__(self, w, beta_w):
        self.beta_w = beta_w
        self.w = w
        self.c0 = C0

    @property
    def Lw(self):
        r"""Frequency-domain representation of nonlinear operator.

        Returns:
            exception `NotImplementedError`
        """
        raise NotImplementedError

    def Nw(self, uw):
        r"""Frequency-domain representation of nonlinear operator.

        Args:
            uw (:obj:`numpy.ndarray`):
                Frequency-domain representation of field at current
                :math:`z`-position.

        Returns:
            exception `NotImplementedError`
        """
        raise NotImplementedError

    def claw(self, *args):
        r"""Conservation law.

        Callback function that can be used to implementing a measurement
        using a user-supplied function.

        Returns:
            None
        """
        return None


class FMAS_THG(ModelBaseClass):
    r"""Forward model for analytic signal including third-harmonic generation.

    Implements the forward model for the analytic signal including
    third-harmonic generation, see  Eq. (17) of Ref.[1].

    References:
        [1] A. Demircan, Sh. Amiranashvili, C. Bree, U. Morgner, G. Steinmeyer,
        Supercontinuum generation by multiple scatterings at a group velocity
        horizon, Opt. Exp. 22 (2014) 3866,
        https://doi.org/10.1364/OE.22.003866.

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Propagation constant.
        chi (:obj:`float`):
            Nonlinear susceptibility (default=1.0).
    """

    def __init__(self, w, beta_w, chi=1.0):
        super().__init__(w, beta_w)
        self.chi = chi

    @property
    def Lw(self):
        r"""Frequency-domain representation of nonlinear operator.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of linear
            operator of the partial differential equation.
        """
        return 1j * self.beta_w

    def Nw(self, uw):
        r"""Frequency-domain representation of nonlinear operator.

        Args:
            uw (:obj:`numpy.ndarray`):
                Frequency-domain representation of field at current
                :math:`z`-position.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of field at
            current :math:`z`-position.
        """
        w, c0, chi, beta_w = self.w, self.c0, self.chi, self.beta_w
        ut = IFT(uw)

        _gam_w = np.divide(
            chi * w * w,
            c0 * c0 * 8.0 * beta_w,
            out=np.zeros(w.size, dtype="float"),
            where=np.abs(beta_w) > 1e-20,
        )

        FT_pfp = lambda x: np.where(w > 0, FT(x), 0j)
        return 1j * _gam_w * FT_pfp((ut + np.conj(ut)) ** 3)

    def claw(self, i, zi, w, uw):
        r"""Conservation law of the propagation model.

        Implements conserved quantity related to the field energy, given by

        .. math::
            C_{\mathcal{E}}(z) = \sum_\omega |\mathcal{E}_\omega(z)|^2.

        Args:
            i (:obj:`int`):
                Index specifying the current :math:`z`-step.
            zi (:obj:`float`):
                Current :math:`z`-value.
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Freuqency domain representation of the current field.

        Returns:
            :obj:`numpy.ndarray`: value of the conserved quantitiy.
        """
        _fac_w = np.divide(
            np.abs(self.beta_w), w, out=np.zeros(w.size, dtype="float"), where=w > 1e-6
        )
        return np.sum(_fac_w * np.abs(uw) ** 2)


class BMCF(ModelBaseClass):
    r"""Bidirectional model for the complex field.

    Implements the bidirectional model for the complex field (BMCF), i.e.  Eq.
    (31) of Ref.[1]. It includes third-harmonic generation and self-steepening
    for interacting forward and backward components of the optical field.

    References:
        [1] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
        propagation equations for ultrashort optical pulses, Phys. Rev. E 10
        (2010) 013812, http://dx.doi.org/10.1103/PhysRevA.82.013812.

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Propagation constant.
        chi (:obj:`float`):
            Nonlinear susceptibility (default=1.0).
    """

    def __init__(self, w, beta_w, chi=1.0):
        super().__init__(w, beta_w)
        self.chi = chi

    @property
    def Lw(self):
        r"""Frequency-domain representation of nonlinear operator.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of linear
            operator of the partial differential equation.
        """
        return 1j * np.abs(self.beta_w)

    def Nw(self, uw):
        r"""Frequency-domain representation of nonlinear operator.

        Args:
            uw (:obj:`numpy.ndarray`):
                Frequency-domain representation of field at current
                :math:`z`-position.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of field at
            current :math:`z`-position.
        """
        w, c0, chi, beta_w = self.w, self.c0, self.chi, self.beta_w
        ut = IFT(uw)

        _gam_w = np.divide(
            chi * w * w,
            c0 * c0 * 8.0 * np.abs(beta_w),
            out=np.zeros(w.size, dtype="float"),
            where=np.abs(beta_w) > 1e-20,
        )

        return 1j * _gam_w * FT((ut + np.conj(ut)) ** 3)

    def claw(self, i, zi, w, uw):
        r"""Conservation law of the propagation model.

        Implements conserved quantity related to the field energy, given by

        .. math::
            C_{\mathcal{E}}(z) = \sum_\omega |\mathcal{E}_\omega(z)|^2.

        Args:
            i (:obj:`int`):
                Index specifying the current :math:`z`-step.
            zi (:obj:`float`):
                Current :math:`z`-value.
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Freuqency domain representation of the current field.

        Returns:
            :obj:`numpy.ndarray`: value of the conserved quantitiy.
        """
        _fac_w = np.divide(
            np.abs(self.beta_w), w, out=np.zeros(w.size, dtype="float"), where=w > 1e-6
        )
        return np.sum(_fac_w * np.abs(uw) ** 2)


class FMAS(ModelBaseClass):
    r"""Forward-model for the analytic signal.

    Implements the forward-model for the analytic signal (FMAS) [1,2]. In
    particular, this model implements Eq. (33) of Ref.[1].

    References:
        [1] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
        propagation equations for ultrashort optical pulses, Phys. Rev. E 10
        (2010) 013812, http://dx.doi.org/10.1103/PhysRevA.82.013812.

        [2] Sh. Amiranashvili, A. Demircan, Ultrashort Optical Pulse
        Propagation in terms of Analytic Signal, Adv. Opt. Tech. 2011 (2011)
        989515, http://dx.doi.org/10.1155/2011/989515.

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Propagation constant.
        chi (:obj:`float`):
            Nonlinear susceptibility (default=1.0).
    """

    def __init__(self, w, beta_w, chi=1.0):
        super().__init__(w, beta_w)
        self.chi = chi

    @property
    def Lw(self):
        r"""Frequency-domain representation of nonlinear operator.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of linear
            operator of the partial differential equation.
        """
        return 1j * self.beta_w

    def Nw(self, uw):
        r"""Frequency-domain representation of nonlinear operator.

        Args:
            uw (:obj:`numpy.ndarray`):
                Frequency-domain representation of field at current
                :math:`z`-position.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of field at
            current :math:`z`-position.
        """
        w, c0, chi, beta_w = self.w, self.c0, self.chi, self.beta_w
        ut = IFT(uw)

        _gamma_w = np.divide(
            3.0 * chi * w * w,
            c0 * c0 * 8.0 * beta_w,
            out=np.zeros(w.size, dtype="float"),
            where=np.abs(beta_w) > 1e-20,
        )

        FT_pfp = lambda x: np.where(w > 0, FT(x), 0j)
        return 1j * _gamma_w * FT_pfp(np.abs(ut) ** 2 * ut)

    def claw(self, i, zi, w, uw):
        r"""Conservation law of the propagation model.

        Implements conserved quantity related to the field energy, given by

        .. math::
            C_{\mathcal{E}}(z) = \sum_\omega |\mathcal{E}_\omega(z)|^2.

        Args:
            i (:obj:`int`):
                Index specifying the current :math:`z`-step.
            zi (:obj:`float`):
                Current :math:`z`-value.
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Freuqency domain representation of the current field.

        Returns:
            :obj:`numpy.ndarray`: value of the conserved quantitiy.
        """
        return np.sum(np.abs(uw[w > 0]) ** 2)


class FMAS_S(ModelBaseClass):
    r"""Simplified variant of the forward-model for the analytic signal.

    Implements a simplified variant of the forward-model for the analytic
    signal [1,2]. In particular, this model implements Eq. (9) of Ref.[2].

    References:
        [1] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
        propagation equations for ultrashort optical pulses, Phys. Rev. E 10
        (2010) 013812, http://dx.doi.org/10.1103/PhysRevA.82.013812.

        [2] A. Demircan, Sh. Amiranashvili, C. Bree, C. Mahnke, F. Mitschke, G.
        Steinmeyer, Rogue wave formation by accelerated solitons at an optical
        event horizon, Appl. Phys. B 115 (2014) 343,
        http://dx.doi.org/10.1007/s00340-013-5609-9

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Propagation constant.
        n2 (:obj:`float`):
            Nonlinear refractive index (default=1.0).
    """

    def __init__(self, w, beta_w, n2):
        super().__init__(w, beta_w)
        self.n2 = n2

    @property
    def Lw(self):
        r"""Frequency-domain representation of nonlinear operator.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of linear
            operator of the partial differential equation.
        """
        return 1j * self.beta_w

    def Nw(self, uw):
        r"""Frequency-domain representation of nonlinear operator.

        Args:
            uw (:obj:`numpy.ndarray`):
                Frequency-domain representation of field at current
                :math:`z`-position.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of field at
            current :math:`z`-position.
        """
        w, c0, n2, beta_w = self.w, self.c0, self.n2, self.beta_w
        ut = IFT(uw)
        _gamma = n2 * w / c0
        FT_pfp = lambda x: np.where(w > 0, FT(x), 0j)
        return 1j * _gamma * FT_pfp(np.abs(ut) ** 2 * ut)

    def claw(self, i, zi, w, uw):
        r"""Conservation law of the propagation model.

        Implements conserved quantity related to the field energy, given by

        .. math::
            C_{\mathcal{E}}(z) = \sum_\omega |\mathcal{E}_\omega(z)|^2.

        Args:
            i (:obj:`int`):
                Index specifying the current :math:`z`-step.
            zi (:obj:`float`):
                Current :math:`z`-value.
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Freuqency domain representation of the current field.

        Returns:
            :obj:`numpy.ndarray`: value of the conserved quantitiy.
        """
        return np.sum(np.abs(uw[w > 0]) ** 2)


class FMAS_S_Raman(ModelBaseClass):
    r"""Simplified variant of the forward-model for the analytic signal
    including the Raman effect.

    Implements a simplified variant of the forward-model for the analytic
    signal including the Raman effect [1]. In particular, this model implements
    Eq. (10) of Ref.[1].

    Aliased as :class:`FMAS_S_R`.

    References:
        [1] A. Demircan, Sh. Amiranashvili, C. Bree, C. Mahnke, F. Mitschke, G.
        Steinmeyer, Rogue wave formation by accelerated solitons at an optical
        event horizon, Appl. Phys. B 115 (2014) 343,
        http://dx.doi.org/10.1007/s00340-013-5609-9

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Propagation constant.
        n2 (:obj:`float`):
            Nonlinear refractive index (default=1.0).
        fR (:obj:`float`):
            Fractional raman response (default=0.18).
        tau1 (:obj:`float`):
            Time scale associated with oscillator angular
            frequency in Lorentz model of Raman response (default=12.2 fs).
        tau2 (:obj:`float`):
            Time scale associated with oscillator angular
            frequency in Lorentz model of Raman response (default=32.0 fs).
    """

    def __init__(self, w, beta_w, n2, fR=0.18, tau1=12.2, tau2=32.0):
        super().__init__(w, beta_w)
        self.n2 = n2
        self.fR = fR
        self.hRw = self._initialize_Raman_response(tau1, tau2)

    def _initialize_Raman_response(self, tau1, tau2):
        r"""Helper function for Raman response.

        Implements simple raman response of type

        .. math::
          h_R(\omega) = \frac{\tau_1^2 + \tau_2^2}
                                {\tau_1^2(1-i\omega \tau_2)^2 + \tau_2^2}.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of Raman
            response.
        """
        _hR_func = lambda w: (tau1 ** 2 + tau2 ** 2) / (
            tau1 ** 2 * (1 - 1j * w * tau2) ** 2 + tau2 ** 2
        )
        return _hR_func(self.w)

    @property
    def Lw(self):
        r"""Frequency-domain representation of nonlinear operator.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of linear
            operator of the partial differential equation.
        """
        return 1j * self.beta_w

    def Nw(self, uw):
        r"""Frequency-domain representation of nonlinear operator.

        Args:
            uw (:obj:`numpy.ndarray`):
                Frequency-domain representation of field at current
                :math:`z`-position.

        Returns:
            :obj:`numpy.ndarray`: Frequency-domain representation of field at
            current :math:`z`-position.
        """
        w, c0, n2, fR, hRw = self.w, self.c0, self.n2, self.fR, self.hRw
        _gamma = n2 * w / c0
        FT_pfp = lambda x: np.where(w > 0, FT(x), 0j)
        N_Raman = lambda u: (1 - fR) * u * np.abs(u) ** 2 + fR * u * IFT(
            FT(np.abs(u) ** 2) * hRw
        )
        return 1j * _gamma * FT_pfp(N_Raman(IFT(uw)))

    def claw(self, i, zi, w, uw):
        r"""Conservation law of the propagation model.

        Implements conserved quantity related to the field mass, given by

        .. math::
            C_{\mathcal{E}}(z)=\sum_{\omega>0}|\mathcal{E}_\omega(z)|^2/\omega.

        Args:
            i (:obj:`int`):
                Index specifying the current :math:`z`-step.
            zi (:obj:`float`):
                Current :math:`z`-value.
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Freuqency domain representation of the current field.

        Returns:
            :obj:`numpy.ndarray`: value of the conserved quantitiy.
        """
        return np.sum(np.abs(uw[w > 0]) ** 2 / w[w > 0])


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


# ALIAS FOR FMAS_S_Raman
FMAS_S_R = FMAS_S_Raman
