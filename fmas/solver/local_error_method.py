"""
Implements local error method (LEM).

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .solver_base import SolverBaseClass
from ..stepper import RungeKutta2, RungeKutta4


class LEM_SySSM(SolverBaseClass):
    r"""Adaptive stepsize local error method (LEM).

    Implements a :math:`z`-propagation scheme with adaptive step size controll,
    referred to as the local error method (LEM) [1].  This method is based on
    the technique of step-doubling, providing a coarse and fine field solution,
    and the assessment of a relative local error (RLE) to guide step size
    selection. If an adequate step size for the current integration sub-step is
    found, a field solution is obtained from the accepted coarse and fine
    solutions through local extrapolation.

    Note:
       *    Local extrapolation yields increased accuracy. In the provided
            implementation, the field update is performed using a symmetric
            split step Fourier method. In itself, this method enables numerical
            schemes with maximal achievale local error
            :math:`\mathcal{O}(h^3)`, where :math:`h` is the step size.  In
            conjunction with a second-order Runge-Kutta formula, a single field
            update based on local extrapolation achieves local error
            :math:`\mathcal O(h^4)`.  In conjunction with a fourth-order
            Runge-Kutta formula, a single field update based on local
            extrapolation can, under fortunate circumstances, even exceed the
            expected :math:`\mathcal O(h^4)` scaling [2].

       *    The true advantage of the LEM is not the apparent higher order, but
            the possibility to control the performance of the algorithm by
            adapting the step size.

       *    In comparison to the number of evaluations of the nonlinear term
            :math:`\mathsf{N}` needed to compute the fine solution, the
            overhead cost of a single local extrapolation is a factor 1.5.

    References:
        [1] O. V. Sinkin, R. Holzlöhner, J. Zweck, C. R. Menyuk,
        Optimization of the split-step Fourier method in modeling optical-fiber
        communications systems,
        IEEE J. Lightwave Tech. 21 (2003) 61,
        https://doi.org/10.1109/JLT.2003.808628.

        [2] A. M. Heidt,
        Efficient Adaptive Step Size Method for the Simulation of
        Supercontinuum Generation in Optical Fibers,
        IEEE J. Lightwave Tech. 27 (2009) 3984,
        https://doi.org/10.1109/JLT.2009.2021538

    Aliased as :class:`LEM`.

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

    def __init__(self, L, N, stepper=RungeKutta2, del_G=1e-5, user_action=None):
        super().__init__(L, N, stepper, user_action=user_action)
        self.del_G = del_G
        self.scale_fac = 1.2599210498948732
        self.dz_a = np.inf
        self._dz_a = []
        self._del_rle = []

    def single_step(self, z_curr, Ew):
        r"""Advance field by a single :math:`z`-slice.

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
        # ... DEFINE RELATIVE LOCAL ERROR (RLE)
        _norm = lambda u: np.sqrt(np.sum(np.abs(u) ** 2))
        _rle = lambda uf, uc: _norm(uf - uc) / _norm(uf)
        # ... DEFINE SYMMETRIC SPLIT-STEP FOURIER METHOD
        _P_lin = lambda z: np.exp(L * z)  # exact linear propagator
        _dEwdz = lambda z, Ew: self.N(Ew)  # nonlinear function

        def _step(Ew, dz):
            return _P_lin(dz / 2) * P(_dEwdz, 0.0, _P_lin(dz / 2) * Ew, dz)

        # ... FIELD UPDATE COMPLETING SUBSTEP
        def _field_update(_len, dz_a, Ew, Ew_trial):
            if _len <= dz_a:
                # -- STEP EXCEEDS CURRENT Z-SLICE
                # ... LIMIT SUBSTEP TO LAND AT UPPER BOUNDARY OF Z-SLICE
                uc = _step(Ew, _len)
                uf = _step(_step(Ew, _len / 2), _len / 2)
                Ew = (4 * uf - uc) / 3
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
            uc = _step(Ew, dz_a)
            uf = _step(_step(Ew, dz_a / 2), dz_a / 2)
            Ew_trial = (4 * uf - uc) / 3
            del_curr = _rle(uf, uc)
            if del_curr > 2 * del_G:
                # ... CASE 1: RELATIVE LOCAL ERROR TOO LARGE
                # ... DISCARD SOLUTION AND RETRY WITH HALVED STEP SIZE
                dz_a *= 0.5
            elif (del_curr > del_G) and (del_curr < 2 * del_G):
                # ... CASE 2: RELATIVE LOCAL ERROR TOO LARGE
                # ... KEEP SOLUTION AND DECREASE STEP SIZE FOR NEXT SUBSTEP
                _len, Ew = _field_update(_len, dz_a, Ew, Ew_trial)
                dz_a /= scale_fac
            elif del_curr < 0.5 * del_G:
                # ... CASE 3: RELATIVE LOCAL ERROR TOO SMALL
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


class LEM_IFM(SolverBaseClass):
    r""":math:`z`-propagation algorithm implementing a modified local-error
    method (LEM-IFM)

    Implements a :math:`z`-propagation scheme with adaptive step size controll,
    based on the local-error method (LEM) [1] in conjunction with a variant of
    the integrating factor method for which the reference position :math:`z_0`
    coincides with as the starting position :math:`z` of each substep.

    Here, the advantage is that when step-doubling, it is possible to exploit
    the fact that the full step, needed to compute the coarse solution, and the
    first half-step, needed to compute the fine solution, have one evaluation
    of the nonlinear function :math:`\mathsf{N}` in common. Thus, a custom
    fourth-order Runge-Kutta scheme can be devised that exploits this
    redundancy and saves one evaluation of the nonlinear function.  In
    comparison to the number of evaluations of the nonlinear term needed to
    compute the fine solution, the overhead cost of a single local
    extrapolation is a factor 1.375.
    This is beneficial in cases where the evaluation of the nonlinear function
    comes at high computational cost, e.g. when the underlying propagation
    model accounts for the Raman effect.

    Note:
       *    Local extrapolation yields increased accuracy. In the provided
            implementation, the field update is performed using a variant if
            the integrating factor method in conjunction with a fourth-order
            Runge-Kutta (RK4) formula. Here, a single field update based on
            local extrapolation achieves local error :math:`\mathcal O(h^6)`
            scaling [2].

       *    As for the standard LEM,
            true advantage of the modified LEM is not the apparent higher
            order, but the possibility to control the performance of the
            algorithm by adapting the step size.

       *    In comparison to the number of evaluations of the nonlinear term
            :math:`\mathsf{N}` needed to compute the fine solution, the
            overhead cost of a single local extrapolation is a factor 1.375.
            This is a result of the coarse step, requiring 4 evaluations of
            :math:`\mathsf{N}`, and the fine step, requiring additional 7
            evaluations of :math:`\mathsf{N}`.
            Compared to the 8 evaluations a sole fine step would need, this
            then amounts to an increase of a factor of 11/8 evaluations of
            :math:`\mathsf{N}`.
            Thus, in comparison to the standard LEM and the LEM based on the
            Runge-Kutta in the interaction picture method [3], the overhead
            cost of step-doubling is about :math:`10\%` smaller.

    References:
        [1] O. V. Sinkin, R. Holzlöhner, J. Zweck, C. R. Menyuk,
        Optimization of the split-step Fourier method in modeling optical-fiber
        communications systems,
        IEEE J. Lightwave Tech. 21 (2003) 61,
        https://doi.org/10.1109/JLT.2003.808628.

        [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
        Numerical Recipes in C: The art of scientific computing (Chapter 16.1),
        Cambridge University Press (1992).

        [3] A. M. Heidt,
        Efficient Adaptive Step Size Method for the Simulation of
        Supercontinuum Generation in Optical Fibers,
        IEEE J. Lightwave Tech. 27 (2009) 3984,
        https://doi.org/10.1109/JLT.2009.2021538

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

    def __init__(self, L, N, del_G=1e-5, user_action=None):
        super().__init__(L, N, stepper=None, user_action=user_action)
        self.del_G = del_G
        self.scale_fac = 1.148698354997035
        self.dz_a = np.inf
        self._dz_a = []
        self._del_rle = []

    def single_step(self, z_curr, Ew):
        r"""Advance field by a single :math:`z`-slice

        Uses a custom fourth-order Runge-Kutta algorithm for step-doubling,
        exploiting the fortunate fact that the full step and the first
        half-step have one evaluation of the nonlinear function
        :math:`\mathsf{N}` in common.

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
        # ... DEFINE RELATIVE LOCAL ERROR (RLE)
        _norm = lambda u: np.sqrt(np.sum(np.abs(u) ** 2))
        _rle = lambda uf, uc: _norm(uf - uc) / _norm(uf)
        # ... DEFINE INTEGRATING FACTOR METHOD WITH REF. DIST. Z0=Z_CURR
        _P_lin = lambda z: np.exp(L * z)

        def _dEIwdz(h, EIw):
            x = _P_lin(h)
            return N(x * EIw) / x

        # _dEIwdz = lambda h, EIw: _P_lin(-h)*N(_P_lin(h)*EIw)
        def _shared_init_step_RK4(fun, z, uw, h):
            # -- SHARED STEP
            k1 = fun(z, uw)
            # -- COARSE SOLUTION
            k2 = fun(z + 0.5 * h, uw + 0.5 * h * k1)
            k3 = fun(z + 0.5 * h, uw + 0.5 * h * k2)
            k4 = fun(z + h, uw + h * k3)
            uw_c = uw + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            uw_c *= _P_lin(h)
            # -- FINE SOLUTION
            h = 0.5 * h
            # ... COMPLETE FIRST HALF-STEP
            k2 = fun(z + 0.5 * h, uw + 0.5 * h * k1)
            k3 = fun(z + 0.5 * h, uw + 0.5 * h * k2)
            k4 = fun(z + h, uw + h * k3)
            tmp = uw + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            tmp *= _P_lin(h)
            # ... SECOND HALF-STEP
            k1 = fun(z, tmp)
            k2 = fun(z + 0.5 * h, tmp + 0.5 * h * k1)
            k3 = fun(z + 0.5 * h, tmp + 0.5 * h * k2)
            k4 = fun(z + h, tmp + h * k3)
            uw_f = tmp + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            uw_f *= _P_lin(h)
            return uw_c, uw_f

        def _step_doubling(Ew, h):
            return _shared_init_step_RK4(_dEIwdz, 0.0, Ew, h)

        # ... FIELD UPDATE COMPLETING SUBSTEP
        def _field_update(_len, dz_a, Ew, Ew_trial):
            if _len <= dz_a:
                # -- STEP EXCEEDS CURRENT Z-SLICE
                # ... LIMIT SUBSTEP TO LAND AT UPPER BOUNDARY OF Z-SLICE
                uc, uf = _step_doubling(Ew, _len)
                Ew = (16 * uf - uc) / 15
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
            uc, uf = _step_doubling(Ew, dz_a)
            Ew_trial = (16 * uf - uc) / 15
            del_curr = _rle(uf, uc)
            if del_curr > 2 * del_G:
                # ... CASE 1: RELATIVE LOCAL ERROR TOO LARGE
                # ... DISCARD SOLUTION AND RETRY WITH HALVED STEP SIZE
                dz_a *= 0.5
            elif (del_curr > del_G) and (del_curr < 2 * del_G):
                # ... CASE 2: RELATIVE LOCAL ERROR TOO LARGE
                # ... KEEP SOLUTION AND DECREASE STEP SIZE FOR NEXT SUBSTEP
                _len, Ew = _field_update(_len, dz_a, Ew, Ew_trial)
                dz_a /= scale_fac
            elif del_curr < 0.5 * del_G:
                # ... CASE 3: RELATIVE LOCAL ERROR TOO SMALL
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
