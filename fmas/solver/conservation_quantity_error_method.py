"""
Implements conservation quantity error method (CQE).

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .solver_base import SolverBaseClass
from ..stepper import RungeKutta4


class CQE_RK4IP(SolverBaseClass):
    r"""Adaptive stepsize conservation quantity error (CQE) method.

    Implements :math:`z`-propagation scheme with adaptive step size control,
    based on the conservation quantity error (CQE) method [1].
    For :math:`z`-propagation, an integrating factor method (IFM) for which the
    reference position :math:`z_0` coincides with as the starting position
    :math:`z` of each substep is used.

    Note:
    * The IFM variant used for :math:`z`-propagation is similar, but not
    identical to the `Runge-Kutta in the interaction picture` (RK4IP) method [2].
    The latter uses the midpoint of the  proposed substep extend as reference
    position.

    References:
        [1] A. M. Heidt,
        Efficient Adaptive Step Size Method for the Simulation of
        Supercontinuum Generation in Optical Fibers,
        IEEE J. Lightwave Tech. 27 (2009) 3984,
        https://doi.org/10.1109/JLT.2009.2021538

        [2] J. Hult, A Fourth-Order Runge–Kutta in the Inter- action Picture
        Method for Simulating Supercontin- uum Generation in Optical Fibers,
        IEEE J. Light- wave Tech. 25 (2007) 3770,
        https://doi.org/10.1109/JLT.2007.909373.

    Args:
        L (:obj:`numpy.ndarray`):
            Linear operator of the partial differential equation.
        N (:obj:`numpy.ndarray`):
            Nonlinear operator of the partial differential equation.
        stepper (:obj:`function`):
            z-stepping algorithm. Default is a 2nd-order Runge-Kutta formula.
        del_G (:obj:`float`): Goal local error (default is `del_G = 1e-5`).

    Attributes:
        del_G (:obj:`float`):
            Goal local error.
        scale_fac (:obj:`float`):
            Step size scaling factor (initialized as `scale_fac = numpy.inf`).
        dz_a (:obj:`float`):
            Local step size.
        _dz_a (:obj:`list` of :obj:`float`):
            Array accumulating local step size at the end of each
            :math:`z`-slice.
        _del_rle (:obj:`list` of :obj:`float`):
            Array accumulating the local relative error at the end of each
            :math:`z`-slice.
    """

    def _default_CQE_fun(i, zi, w, uw):
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

    def __init__(self, L, N, del_G=1e-5, user_action=_default_CQE_fun):
        super().__init__(L, N, stepper=RungeKutta4, user_action=user_action)
        self.del_G = del_G
        self.scale_fac = 1.148698354997035
        self.dz_a = np.inf
        self._dz_a = []
        self._del_rle = []

    def single_step(self, z_curr, Ew):
        r"""Advance field by a single :math:`z`-slice.

        To advance the field of a single :math:`z`-step, this method might
        perform several substeps of smaller size.

        Note:
            This method updates the instance attributes `_dz_a` and `_del_rle`,
            i.e.  lists accumulating the local step size at the end of each
            :math:`z`-slize, and the associated relative local error,
            respectively.

        Args:
            z_curr (:obj:`float`): Current propagation distance.
            Ew (:obj:`numpy.ndarray`): Frequency domain representation of the
                field at `z_curr`.

        Returns:
            :obj:`numpy.ndarray`: Frequency domain representation of the field
            at `z_curr` + `dz`.

        """
        # -- STRIP OFF INSTANCE KEYWORD
        dz, w, L, N, P = self.dz_, self.w, self.L, self.N, self.stepper
        dz_a, del_G, scale_fac = self.dz_a, self.del_G, self.scale_fac

        # -- FUNCTION DEFINITIONS AND INITIALIZATION
        # ... INITIALIZE ADAPTIVE STEPSIZE
        if self._dz_a == []:
            self._dz_a = [dz]
            self._del_rle = [0.0]
        # ... CONSERVATION QUANTITY ERROR FUNCTION
        CQE_fun = lambda Ew: self.ua_fun(0, 0, w, Ew)

        def _CQE(Ew, Ew_trial):
            Cp_ini = CQE_fun(Ew)
            Cp_fin = CQE_fun(Ew_trial)
            return np.abs(Cp_fin - Cp_ini) / Cp_ini

        # ... DEFINE INTEGRATING FACTOR METHOD WITH REF. DIST. Z0=Z_CURR
        _P_lin = lambda z: np.exp(L * z)

        def _dEIwdz(h, EIw):
            _ = _P_lin(h)
            return N(_ * EIw) / _

        def _RK4(fun, Ew, h):
            return _P_lin(h) * self.stepper(fun, 0.0, Ew, h)

        # ... TRIAL STEP RETURNING UPDATED FIELD AND CQE-VALUE
        def _trial_step(Ew, h):
            Ew_trial = _RK4(_dEIwdz, Ew, h)
            del_curr = _CQE(Ew, Ew_trial)
            return Ew_trial, del_curr

        # ... FIELD UPDATE COMPLETING SUBSTEP
        def _field_update(_len, dz_a, Ew, Ew_trial):
            if _len <= dz_a:
                # -- STEP EXCEEDS CURRENT Z-SLICE
                # ... LIMIT SUBSTEP TO LAND AT UPPER BOUNDARY OF Z-SLICE
                Ew, _ = _trial_step(Ew, _len)
                # ... TERMINATE WHILE-LOOP ON NEXT ITERATION ATTEMPT
                _len = -1.0
            else:
                # -- STEP REMAINS INSIDE CURRENT Z-SLICE
                Ew = Ew_trial
                _len -= dz_a
            return _len, Ew

        # -- BEGIN: PERFORM FULL STEP COVERING A SINGLE Z-SLICE
        dz_a = self._dz_a[-1]  # CURRENT STEP SIZE
        _len = dz  # FULL Z-SLICE LENGTH
        while _len > 0.0:
            # -- BEGIN: SUBSTEP
            Ew_trial, del_curr = _trial_step(Ew, dz_a)
            if del_curr > 2 * del_G:
                # ... CASE 1: CQE-VAlUE WAY TOO LARGE
                # ... DISCARD SOLUTION AND RETRY WITH HALVED STEP SIZE
                dz_a *= 0.5
            elif (del_curr > del_G) and (del_curr < 2 * del_G):
                # ... CASE 2: CQE-VALUE  TOO LARGE
                # ... KEEP SOLUTION AND DECREASE STEP SIZE FOR NEXT SUBSTEP
                _len, Ew = _field_update(_len, dz_a, Ew, Ew_trial)
                dz_a /= scale_fac
            elif del_curr < 0.1 * del_G:
                # ... CASE 3: CQE-VALUE TOO SMALL
                # ... KEEP SOLUTION AND INCREASE STEP SIZE FOR NEXT SUBSTEP
                _len, Ew = _field_update(_len, dz_a, Ew, Ew_trial)
                dz_a *= scale_fac
            else:
                _len, Ew = _field_update(_len, dz_a, Ew, Ew_trial)
            # -- END: SUBSTEP
        # -- END: PERFORM FULL STEP COVERING A SINGLE Z-SLICE
        self._del_rle.append(del_curr)
        self._dz_a.append(dz_a)
        return Ew

    def clear(self):
        r"""Clear instance attributes and reset parameters to initial values

        Note:
            This method is implemented for the case when an instance of the
            solver is used for various simulation runs with possibly different
            initial conditions and :math:`z`-interval discretizations.
        """
        # -- PREPARE NEXT RUN OF THE SOLVER BY ...
        # ... CLEARING INTERNAL DATA ACCUMULATING STRUCTURES OF SUPERCLASS
        super().clear()
        # ... RESETTING VARIABLE STEPSIZE TO ITS INITIAL VALUE
        self.dz_a = np.inf
