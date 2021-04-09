r"""
Soliton implosion in a NLPM750 PCF
==================================

This examples demonstrates breakdown of a high-order soliton in a NL-PM-750
photonic crystal fiber, described in Ref. [1].  For the numerical simualtion
the forward model for the analytic signal including the Raman effect is used
[1,2].

In particular, this example reproduces the propagation scenario  shown in
Fig.~2 of Ref. [1]. The resulting figure shows the breakdown of a high-order
soliton of duration :math:`t_{\rm{S}}=10\,\mathrm{fs}`, center frequency
:math:`\omega_{\rm{S}}=1.884\,\mathrm{rad/fs}`, and soliton order
:math:`N_{\rm{S}}=10`. This process is also referred to as soliton implosion.
For more details see Ref. [1].

References:
    [1] I. Babushkin, A. Tajalli, H. Sayinc et al., Simple route toward
    efficient frequency conversion for generation of fully coherent
    supercontinua in the mid-IR and UV range, Light: Science & Applications 6
    (2017) e16218, https://doi.org/10.1038/lsa.2016.218

    [2] A. Demircan, Sh. Amiranashvili, C. Bree, C. Mahnke, F. Mitschke, G.
    Steinmeyer, Rogue wave formation by accelerated solitons at an optical
    event horizon, Appl. Phys. B 115 (2014) 343,
    http://dx.doi.org/10.1007/s00340-013-5609-9

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import fmas
import numpy as np
from fmas.grid import Grid
from fmas.models import FMAS_S_R
from fmas.solver import IFM_RK4IP
from fmas.analytic_signal import AS
from fmas.propagation_constant import PropConst, define_beta_fun_NLPM750
from fmas.tools import change_reference_frame, plot_evolution


def main():

    # -- INITIALIZATION STAGE
    # ... DEFINE SIMULATION PARAMETERS
    t_max = 3000.       # (fs)
    t_num = 2**14       # (-)
    z_max = 8.0e3       # (micron)
    z_num = 10000       # (-)
    z_skip = 10         # (-)
    n2 = 3.0e-8         # (micron^2/W)
    wS = 1.884          # (rad/fs)
    tS = 10.0           # (fs)
    NS = 10.            # (-)
    # ... PROPAGGATION CONSTANT
    beta_fun = define_beta_fun_NLPM750()
    pc = PropConst(beta_fun)
    # ... COMPUTATIONAL DOMAIN, MODEL, AND SOLVER 
    grid = Grid( t_max = t_max, t_num = t_num, z_max = z_max, z_num = z_num)
    model = FMAS_S_R(w=grid.w, beta_w=pc.beta(grid.w), n2 = n2)
    solver = IFM_RK4IP( model.Lw, model.Nw)

    # -- SET UP INITIAL CONDITION
    A0 = NS*np.sqrt(np.abs(pc.beta2(wS))*model.c0/wS/n2)/tS
    Eps_0w = AS(np.real(A0/np.cosh(grid.t/tS)*np.exp(1j*wS*grid.t))).w_rep
    solver.set_initial_condition( grid.w, Eps_0w)

    # -- PERFORM Z-PROPAGATION
    solver.propagate( z_range = z_max, n_steps = z_num, n_skip = z_skip)

    # -- SHOW RESULTS
    utz = change_reference_frame(solver.w, solver.z, solver.uwz, pc.vg(wS))
    plot_evolution( solver.z, grid.t, utz,
        t_lim = (-100,100), w_lim = (0.5,8.), DO_T_LOG = True)


if __name__=='__main__':
    main()
