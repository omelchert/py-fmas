r"""
FMAS_S
======

This examples demonstrates photon number conservation for the simplified
forward model for the analytic signal (FMAS-S).

The considered propagation model provides a proper conservation law as class
method `claw`. However, for clarity, we here re-implement the conservation
law and explicitly pass this  user-defined function to the solver class upon
initialization.

As exemplary propagation scenario, the setup used in the step-by-step demo

:ref:`sphx_glr_auto_tutorials_specific_g_spectrogram.py`

is used.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import fmas
import numpy as np
from fmas.solver import IFM_RK4IP
from fmas.analytic_signal import AS
from fmas.grid import Grid
from fmas.propagation_constant import PropConst, define_beta_fun_ESM
from fmas.tools import sech, change_reference_frame, plot_photon_number

beta_fun = define_beta_fun_ESM()
pc = PropConst(beta_fun)

grid = Grid(t_max=5500.0, t_num=2 ** 14)  # (fs)  # (-)

Ns = 8.0  # (-)
t0 = 7.0  # (fs)
w0 = 1.7  # (rad/fs)
n2 = 3.0e-8  # (micron^2/W)
A0 = Ns * np.sqrt(abs(pc.beta2(w0)) * pc.c0 / w0 / n2) / t0
E_0t_fun = lambda t: np.real(A0 * sech(t / t0) * np.exp(1j * w0 * t))
Eps_0w = AS(E_0t_fun(grid.t)).w_rep

###############################################################################
# As model we here consider the simplified forward model for the analytic
# signal (FMAS_S) 

from fmas.models import FMAS_S
model = FMAS_S(w=grid.w, beta_w=pc.beta(grid.w), n2=n2)

###############################################################################
# For the FMAS_S :math:`z`-propagation model we consider a conserved quantity
# that is related to the classical analog of the photon number, see Eq. (24) of
# [AD2010] and the appendix of [BW1989]. In particular we here implement
#
# .. math::
#    C_p(z) = \sum_{\omega>0} \omega^{-1} |u_\omega(z)|^2,
#
# which is, by default, provided as method `model.claw` .


def Cp(i, zi, w, uw):
    _a2_w = np.divide(
        np.abs(uw) ** 2, w, out=np.zeros(w.size, dtype="float"), where=w >0.
    )
    return np.sum(_a2_w)


###############################################################################
# As shown below, this conserved quantity can be provided when an instance of
# the desired solver is initialized. Here, for simply monitoring the
# conservation law we use the Runge-Kutta in the ineraction picture method.
# However, a proper conserved quantity is especially important when the
# conservation quantity error method (CQE) is used, see, e.g., demo
#
# :ref:`sphx_glr_auto_tutorials_tests_g_performance_CQE.py`
#

solver = IFM_RK4IP(model.Lw, model.Nw, user_action=Cp)
solver.set_initial_condition(grid.w, Eps_0w)
solver.propagate(z_range=0.01e6, n_steps=4000, n_skip=8)  # (micron)  # (-)  # (-)

###############################################################################
# The figure below shows the dynamic evolution of the pulse in the time domain
# (top subfigure) and in the frequency domain (center subfigure). The subfigure
# at the bottom shows the normalized photon number variation
#
# .. math::
#    \delta_{\rm{Ph}}(z) = \frac{ C_p(z)-C_p(0)}{C_p(0)}
#
# as function of the proapgation coordinate :math:`z`. For the considered
# discretization of the computational domain the normalized photon number
# variation is of the order :math:`\delta_{\rm{Ph}}\approx 10^{-7}` and thus
# very small. The value can be still decreased by decreasing the stepsize
# :math:`\Delta z`.

utz = change_reference_frame(solver.w, solver.z, solver.uwz, pc.vg(w0))

plot_photon_number(
    solver.z, grid.t, utz, solver.ua_vals, t_lim=(-25, 125), w_lim=(0.5, 4.5)
)

###############################################################################
# **References:**
#
# [AD2010] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of propagation
# equations for ultrashort optical pulses, Phys. Rev. E 10 (2010) 013812,
# http://dx.doi.org/10.1103/PhysRevA.82.013812.
#
# [BW1989] K. J. Blow, D. Wood, Theoretical description of transient stimulated
# Raman scattering in optical fibers.  IEEE J. Quantum Electron., 25 (1989)
# 1159, https://doi.org/10.1109/3.40655.
