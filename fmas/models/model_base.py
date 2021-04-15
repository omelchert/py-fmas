"""
Implements model bas class.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from ..config import FTFREQ, FT, IFT, C0


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
