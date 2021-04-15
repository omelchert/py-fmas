"""
Implements bidirectional model for the complex field.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .model_base import ModelBaseClass
from ..config import FTFREQ, FT, IFT, C0


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
            np.abs(self.beta_w) * np.abs(uw) ** 2,
            w * w,
            out=np.zeros(w.size, dtype="float"),
            where=w > 1e-6,
        )
        return np.sum(_fac_w)
