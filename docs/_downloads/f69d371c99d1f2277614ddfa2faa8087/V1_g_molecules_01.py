r"""
Generation of two-frequency soliton moleculs - 1
================================================

This examples demonstrates the generation of two-frequency soliton molecules,
using the forward model for the analytic signal [1,2], in `py-fmas` implemented
as :class:`FMAS`.

In particular, this example shows how soliton molecules are generated from two
initially superimposed fundamental solitons at distinctly different
frequencies [3].

References:

    [1] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
    propagation equations for ultrashort optical pulses, Phys. Rev. E 10
    (2010) 013812, http://dx.doi.org/10.1103/PhysRevA.82.013812.

    [2] Sh. Amiranashvili, A. Demircan, Ultrashort Optical Pulse Propagation in
    terms of Analytic Signal, Adv. Opt. Tech. 2011 (2011) 989515,
    http://dx.doi.org/10.1155/2011/989515.

    [3] O. Melchert, S. Willms, S. Bose, A. Yulin, B. Roth, F. Mitschke, U.
    Morgner, I. Babushkin, A. Demircan, Soliton Molecules with Two Frequencies,
    Phys. Rev. Lett. 123 (2019) 243905,
    https://doi.org/10.1103/PhysRevLett.123.243905.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import sys; sys.path.append('../../')
import fmas
import numpy as np
from fmas.models import FMAS
from fmas.config import FTFREQ, FT, IFT, C0
from fmas.solver import IFM_RK4IP
from fmas.analytic_signal import AS
from fmas.grid import Grid
from fmas.tools import plot_evolution


def beta_details():
    # -- EXPANSION COEFFIENTS
    coeffs = [-0.7/24., 0.0, 0.1/2, 13.0, 25.0]
    # -- FREQUENCY FOR WHICH EXPANSION COEFFICIENTS ARE VALID
    w_e = 2.      # (rad/fs)
    # -- PROPAGATION CONSTANT (1/micron)
    _beta = np.poly1d(coeffs); _beta_s = lambda w: _beta(w-w_e)
    # -- GROUP-DELAY PROFILE (fs/micron)
    _beta1 = np.polyder(_beta,m=1); _beta1_s = lambda w: _beta1(w-w_e)
    # -- GROUP-VELOCITY DISPERSION PROFILE (fs^2/micron)
    _beta2 = np.polyder(_beta,m=2); _beta2_s = lambda w: _beta2(w-w_e)
    return _beta_s, _beta1_s, _beta2_s


def main():

    t_max = 4000.       # (fs)
    t_num = 2**14       # (-)
    t = np.linspace(-t_max, t_max, t_num, endpoint=False)
    w = FTFREQ(t.size,d=t[1]-t[0])*2*np.pi
    # -- DEFINE SIMULATION PARAMETERS
    chi = 1.0
    beta, beta1, beta2 = beta_details()
    gamma = lambda w: 3.*chi*w*w/C0/C0/beta(w)/8.0

    w_ZDW1 = 1.46547752252
    w_ZDW2 = 2.53452247748
    w_GVM1 = 1.07417956702
    w_GVM2 = 2.92582137713

    w0 = w_GVM1
    t0 = 20.
    A0 = np.sqrt(np.abs(beta2(w0))/gamma(w0))/t0
    At1 = np.real(A0/np.cosh(t/t0)*np.exp(1j*w0*t))

    w0 = w_GVM2
    t0 = 20.
    A0 = np.sqrt(np.abs(beta2(w0))/gamma(w0))/t0
    At2 = np.real(A0/np.cosh(t/t0)*np.exp(1j*w0*t))

    #for i in range(w.size):
    #    print( w[i], beta2(w[i])  )
    #print (beta2(w_GVM1), beta1(w_GVM1))
    #print (beta2(w_GVM2), beta1(w_GVM2))
    #exit()

    LD = t0*t0/abs(beta2(w_GVM1))

    E_0t = At1 + At2

    # ... COMPUTATIONAL DOMAIN
    z_max = 15*LD    # (micron)
    z_num = 15000        # (-)
    z_skip = 50         # (-)
    # ... INITIAL CONDITION

    # -- INITIALIZATION STAGE
    # ... COMPUTATIONAL DOMAIN
    grid = Grid( t_max = t_max, t_num = t_num, z_max = z_max, z_num = z_num)
    # ... CUSTOM PROPAGATION MODEL
    model = FMAS(w=grid.w, beta_w = beta(w) , chi = chi )
    # ... PROPAGATION ALGORITHM 
    solver = IFM_RK4IP( model.Lw, model.Nw, user_action = model.claw)
    solver.set_initial_condition( grid.w, AS(E_0t).w_rep)

    # -- RUN SIMULATION
    solver.propagate( z_range = z_max, n_steps = z_num, n_skip = z_skip)

    v0 = 1./beta1(w_GVM1)
    utz = IFT(solver.uwz*np.exp(-1j*w*solver.z[:,np.newaxis]/v0), axis=-1)

    # -- SHOW RESULTS
    plot_evolution( solver.z, grid.t, utz,
        t_lim = (-1000,1000), w_lim = (0.3,3.8))


if __name__=='__main__':
    main()
