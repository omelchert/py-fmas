"""
Implements simplified forward model for the analytic signal.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .model_base import ModelBaseClass
from ..config import FTFREQ, FT, IFT, C0


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
        alpha_w (:obj:`numpy.ndarray`):
            Frequency-domain representation of root-power loss.
        n2 (:obj:`float`):
            Nonlinear refractive index (default=1.0).
    """

    def __init__(self, w, beta_w, n2=1.0, alpha_w=0.0):
        super().__init__(w, beta_w, alpha_w)
        self.n2 = n2

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
        return np.sum(np.abs(uw[w > 0]) ** 2 / w[w > 0])
