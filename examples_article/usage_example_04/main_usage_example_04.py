import sys
import numpy as np
from fmas.models import ModelBaseClass
from fmas.config import FTFREQ, FT, IFT, C0
from fmas.grid import Grid
from fmas.solver import LEM
from fmas.data_io import save_h5

class NSE(ModelBaseClass):

    def __init__(self, w, beta, gamma):
        super().__init__(w, beta_w=beta)
        self.gamma = gamma

    @property
    def Lw(self):
        return 1j*self.beta_w

    def N(self, uw):
        ut = IFT(uw)
        return 1j*self.gamma*FT(np.abs(ut)**2*ut)


def main():

    # -- SET MODEL PARAMETERS
    t_max = -50.
    Nt = 2**12
    # ... PROPAGATION CONSTANT (POLYNOMIAL MODEL)
    beta = np.poly1d([-0.5, 0.0, 0.0])
    beta1 = np.polyder(beta, m=1)
    beta2 = np.polyder(beta, m=2)
    # ... NONLINEAR PARAMETER 
    gamma = 1.
    # ... SOLITON PARAMTERS
    t0 = 1.                             # duration
    t_off = 20.                         # temporal offset
    w0 = 25.                            # detuning
    P0 = np.abs(beta2(0))/t0/t0/gamma   # peak-intensity
    LD = t0*t0/np.abs(beta2(0))         # dispersion length
    # ... EXACT SOLUTION
    u_exact = lambda z, t: np.sqrt(P0)*np.exp(0.5j*gamma*P0*z)/np.cosh(t/t0)


    # -- INITIALIZATION STAGE
    # ... COMPUTATIONAL DOMAIN
    grid = Grid(t_max=t_max, t_num=Nt)
    t, w = grid.t, grid.w
    # ... NONLINEAR SCHROEDINGER EQUATION 
    model = NSE(w, beta(w), gamma)
    # ... PROPAGATION ALGORITHM
    solver = LEM(model.Lw, model.N, del_G = 1e-7)
    # ... INITIAL CONDITION
    u0_t  = u_exact(0.0, t+t_off)*np.exp(1j*w0*t)
    u0_t += u_exact(0.0, t-t_off)*np.exp(-1j*w0*t)
    solver.set_initial_condition(w, FT(u0_t))

    # -- RUN SOLVER 
    solver.propagate(
        z_range = 0.5*np.pi*LD,     # propagation range
        n_steps = 2**9,
        n_skip = 2
    )

    # -- STORE RESULTS
    # ... PREPARE DATA DICTIONARY FOR OUTPUT FILE
    results = {
        't': t,
        'z': solver.z,
        'w': solver.w,
        'u': solver.utz,
        'dz_integration': solver.dz_,
        'dz_a': np.asarray(solver._dz_a),
        'del_rle' : np.asarray(solver._del_rle)
    }
    # ... STORE DATA
    save_h5( './res_LEM_SolSolCollision.h5', **results)


if __name__=='__main__':
    main()
