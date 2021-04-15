"""
Implements split-step Fourier methods.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .solver_base import SolverBaseClass
from ..config import W_MAX_FAC
from ..stepper import RungeKutta2, RungeKutta4


class SySSM(SolverBaseClass):
    r"""Fixed stepsize algorithm implementing the symmetric split step
    method (SySSM).


    Implements a fixed stepsize algorithm referred to as the symmetric split
    step Fourier method (SySSM) as discussed in [1,2].  In itself, this method
    enables numerical schemes with maximal achievale global error
    :math:`\mathcal{O}(\Delta z^2)`, where :math:`\Delta z` is the step size.
    The default :math:`z`-stepper initialized with the SySSM is a fourth-order
    Runge-Kutta formula.

    References:
        [1] P. L. DeVries,
        Application of the Split Operator Fourier Transform method to the
        solution of the nonlinear Schrödinger equation,
        AIP Conference Proceedings 160, 269 (1987),
        https://doi.org/10.1063/1.36847.

        [2] J. Fleck, J. K. Morris, M. J. Feit,
        Time-dependent propagation of high-energy laser beams through the
        atmosphere: II,
        Appl. Phys. 10, (1976) 129,
        https://doi.org/10.1007/BF00882638.

    Args:
        L (:obj:`numpy.ndarray`):
            Linear operator of the partial differential equation.
        N (:obj:`numpy.ndarray`):
            Nonlinear operator of the partial differential equation.
        stepper (:obj:`function`):
            z-stepping algorithm. Default is a 4th-order Runge-Kutta formula.

    """

    def __init__(self, L, N, stepper=RungeKutta4, user_action=None):
        super().__init__(L, N, stepper, user_action)

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
        _cleanup = lambda Ew: np.where(np.abs(w) < W_MAX_FAC * w.max(), Ew, 0j)
        _P_lin = lambda z: np.exp(L * z)
        _dEwdz = lambda z, Ew: self.N(Ew)
        return _P_lin(dz / 2) * P(_dEwdz, 0.0, _P_lin(dz / 2) * Ew, dz)


class SiSSM(SolverBaseClass):
    r"""Fixed stepsize algorithm implementing the simple split step
    method (SiSSM).

    Implements a fixed stepsize algorithm referred to as the simple split step
    Fourier method (SiSSM) as discussed in [1].  In itself, this method enables
    numerical schemes with maximal achievale global error
    :math:`\mathcal{O}(\Delta z)`, where :math:`\Delta z` is the step size.
    The default :math:`z`-stepper initialized with the SiSSM is a second-order
    Runge-Kutta formula.

    References:
        [1] T. R. Taha, M. J. Ablowitz,
        Analytical and numerical aspects of certain nonlinear evolution
        equations. II. Numerical, nonlinear Schrödinger equation,
        J. Comput. Phys. 55 (1984) 203,
        https://doi.org/10.1016/0021-9991(84)90003-2.

    Args:
        L (:obj:`numpy.ndarray`):
            Linear operator of the partial differential equation.
        N (:obj:`numpy.ndarray`):
            Nonlinear operator of the partial differential equation.
        stepper (:obj:`function`):
            z-stepping algorithm. Default is a 2nd-order Runge-Kutta formula.

    """

    def __init__(self, L, N, stepper=RungeKutta2, user_action=None):
        super().__init__(L, N, stepper, user_action)

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
        _cleanup = lambda Ew: np.where(np.abs(w) < W_MAX_FAC * w.max(), Ew, 0j)
        _P_lin = lambda z: np.exp(L * z)
        _dEwdz = lambda z, Ew: self.N(Ew)
        return _P_lin(dz) * P(_dEwdz, 0.0, Ew, dz)
