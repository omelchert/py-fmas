"""
Implements solver base class that serves as a driver for the implemented
:math:`z`-propagation algorithms.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from ..config import FTFREQ, FT, IFT, W_MAX_FAC
from ..tools import ProgressBar
from ..stepper import RungeKutta4


class SolverBaseClass:
    r"""Base class for solver

    Implements solver base class that serves as driver for the implemented
    :math:`z`-propagation algorithms.

    Attributes:
        L (:obj:`numpy.ndarray`):
            Linear operator of the partial differential equation.
        N (:obj:`numpy.ndarray`):
            Nonlinear operator of the partial differential equation.
        stepper (:obj:`function`):
            :math:`z`-stepping algorithm. Default is a fourth-order Runge-Kutta
            formula.
        _z (:obj:`list`):
            :math:`z`-values for which field is stored and available after
            propagation.
        _uwz (:obj:`list`):
            Frequency domain representation of the field at :math:`z`-values
            listed in `_z`.
        w (:obj:`list`):
            Angular frequency mesh.
        ua_fun (:obj:`function`):
            User supplied function.
        ua_vals (:obj:`list` of :obj:`object`):
            List holding return-values of `ua_fun` for each stored `z`-slice.

    Args:
        L (:obj:`numpy.ndarray`):
            Linear operator of the partial differential equation.
        N (:obj:`numpy.ndarray`):
            Nonlinear operator of the partial differential equation.
        stepper (:obj:`function`):
            :math:`z`-stepping algorithm. Default is a 4th-order Runge-Kutta formula.
        user_action (:obj:`function`):
            callback function implementing a measurement using a user-supplied
            function with function call signature `user_action(i, zi, w, uw)`,
            where the arguments are:

            - i (:obj:`int`): Index specifying the current :math:`z`-step.
            - zi (:obj:`float`): Current :math:`z`-value.
            - w (:obj:`numpy.ndarray`): Angular frequency mesh.
            - uw (:obj:`numpy.ndarray`): Freuqency domain representation of the
              current fiels.

    """

    def __init__(self, L, N, stepper=RungeKutta4, user_action=None):
        self.L = L
        self.N = N
        self.w = None
        self._z = []
        self._uwz = []
        self.stepper = stepper
        self.ua_fun = user_action
        self.ua_vals = []

    def set_initial_condition(self, w, uw, z0=0.0):
        r"""Set initial condition

        Args:
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Initial field.
            z0 (:obj:`float`):
                :math:`z`-position of initial field (default is z0 = 0.0).
        """
        self.w = w
        self._uwz.append(uw)
        self._z.append(z0)

    def propagate(self, z_range, n_steps, n_skip=0):
        r"""Propagate field

        Args:
            z_range (:obj:`float`):
                Propagation range.
            n_steps (:obj:`int`):
                Number of integration steps.
            n_skip (:obj:`int`):
                Number of intermediate fiels to skip in output file (default is
                n_skip = 0).
        """
        w, ua_fun = self.w, self.ua_fun
        # -- INITIALIZE Z-SLICES
        z_, self.dz_ = np.linspace(0, z_range, n_steps + 1, retstep=True)
        #pb = ProgressBar(num_iter=z_.size - 1, bar_len=60)
        uw = self._uwz[0]
        if ua_fun is not None:
            self.ua_vals.append(ua_fun(0, z_[0], w, uw))
        # -- SOLVE FOR SUBSEQUENT Z-SLICES
        for i in range(1, z_.size):
            uw = self.single_step(z_[i], uw)
            if i % n_skip == 0:
                self._uwz.append(uw)
                self._z.append(z_[i])
                if ua_fun is not None:
                    self.ua_vals.append(ua_fun(i, z_[i], w, uw))
            #pb.update(i)
        #pb.finish()

    @property
    def utz(self):
        r""":obj:`numpy.ndarray`, 2-dim: Time-domain representation of field"""
        return IFT(np.asarray(self._uwz), axis=-1)

    @property
    def uwz(self):
        r""":obj:`numpy.ndarray`, 2-dim: Frequency-domain representation of
        field"""
        return np.asarray(self._uwz)

    @property
    def z(self):
        r""":obj:`numpy.ndarray`, 1-dim: :math:`z`-slices at which field is
        stored"""
        return np.asarray(self._z)

    def clear(self):
        r"""Clear internal arrays"""
        del self._z
        self._z = []
        del self._uwz
        self._uwz = []

    def single_step(self):
        r"""Advance field by a single :math:`z`-slice"""
        raise NotImplementedError
