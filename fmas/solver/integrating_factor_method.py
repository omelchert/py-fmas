"""
Implements integrating factor method (IFM).

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from ..config import W_MAX_FAC
from ..stepper import RungeKutta4
from .solver_base import SolverBaseClass


class IFM(SolverBaseClass):
    r"""Fixed stepsize algorithm implementing the integrating factor
    method (IFM).

    Implements a fixed stepsize algorithm referred to as the integrating factor
    method as discussed in [1,2]. As reference position, when updating the
    field from :math:`z` to :math:`z + \Delta z`, the provided implementation
    considers the current step midpoint :math:`z_0=z+\Delta z /2`.  The
    :math:`z`-stepper initialized with the IFM is a fourth-order Runge-Kutta
    method. This variant of the IFM achieves global error
    :math:`\mathcal{O}(\Delta z^4)`.

    Note:
        This variant of the IFM is also referred to as the Runge-Kutta in the
        interaction picture (RK4IP) method [3].

    Args:
        L (:obj:`numpy.ndarray`):
            Linear operator of the partial differential equation.
        N (:obj:`numpy.ndarray`):
            Nonlinear operator of the partial differential equation.
        user_action (:obj:`function`): callback function implementing a
            measurement using a user-supplied function with function call
            signature `user_action(i, zi, w, uw)`, where the arguments are:

            * i (:obj:`int`): Index specifying the current :math:`z`-step.

            * zi (:obj:`float`): Current :math:`z`-value.

            * w (:obj:`numpy.ndarray`): Angular frequency mesh.

            * uw (:obj:`numpy.ndarray`): Freuqency domain representation of the
              current fiels.

    Aliased as :class:`IFM_RK4IP`.

    References:
        [1] A.-K. Kassam, L. N. Trefethen,
        Fourth-order time- stepping for stiff PDEs,
        SIAM J. Sci. Comp. 26 (2005) 1214,
        https://doi.org/10.1137/S1064827502410633.

        [2] L. N. Trefethen,
        Spectral Methods in Matlab,
        SIAM, Philadelphia, 2000,
        https://people.maths.ox.ac.uk/trefethen/spectral.html
        (accessed 2021-03-18).

        [3] J. Hult, A Fourth-Order Runge–Kutta in the Inter- action Picture
        Method for Simulating Supercontin- uum Generation in Optical Fibers,
        IEEE J. Light- wave Tech. 25 (2007) 3770,
        https://doi.org/10.1109/JLT.2007.909373.
    """

    def __init__(self, L, N, user_action=None):
        super().__init__(L, N, stepper=RungeKutta4, user_action=user_action)

    def single_step(self, z_curr, Ew):
        r"""Advance field by a single :math:`z`-slice

        Args:
            z_curr (:obj:`float`): Current propagation distance.
            Ew (:obj:`numpy.ndarray`): Frequency domain representation of the
            field at `z_curr`.

        Returns:
            :obj:`numpy.ndarray`: Frequency domain representation of the field
            at `z_curr` + `dz`.
        """
        dz, w, L, N, P = self.dz_, self.w, self.L, self.N, self.stepper
        z0 = z_curr + dz / 2
        _cleanup = lambda Ew: np.where(np.abs(w) < W_MAX_FAC * w.max(), Ew, 0j)
        _P_lin = lambda z: np.exp(L * z)
        _dEIwdz = lambda z, EIw: _P_lin(z0 - z) * N(_P_lin(z - z0) * EIw)
        return _P_lin(dz / 2) * P(_dEIwdz, z_curr, _P_lin(dz / 2) * Ew, dz)
