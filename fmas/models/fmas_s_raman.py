"""
Implements simplified forward model for the analytic signal including the Raman
effect.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .model_base import ModelBaseClass
from ..config import FTFREQ, FT, IFT, C0


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
        alpha_w (:obj:`numpy.ndarray`):
            Frequency-domain representation of root-power loss.
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

    def __init__(self, w, beta_w, alpha_w=0.0, n2=1.0, fR=0.18, tau1=12.2, tau2=32.0):
        super().__init__(w, beta_w, alpha_w)
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
        return 1j * self.beta_w - self.alpha_w

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
