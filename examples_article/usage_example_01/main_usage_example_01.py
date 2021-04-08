import sys
import numpy as np
import numpy.fft as nfft
from fmas.models import ModelBaseClass
from fmas.config import FTFREQ, FT, IFT, C0
from fmas.solver import SiSSM, SySSM, IFM_RK4IP, LEM_SySSM, LEM_IFM
from fmas.stepper import RungeKutta2, RungeKutta4


class NSE(ModelBaseClass):

    def __init__(self, w, b2, gamma):
        super().__init__(w, 0.5*b2*w*w)
        self.gamma = gamma

    @property
    def Lw(self):
        return 1j*self.beta_w

    def Nw(self, uw):
        ut = IFT(uw)
        return 1j*self.gamma*FT(np.abs(ut)**2*ut)


def determine_error(mode, stepper):

    # -- SET MODEL PARAMETERS
    # ... PROPAGATION CONSTANT (POLYNOMIAL MODEL)
    beta = np.poly1d([-0.5, 0.0, 0.0])
    # ... GROUP VELOCITY
    beta1 = np.polyder(beta, m=1)
    # ... GROUP VELOCITY DISPERSION
    beta2 = np.polyder(beta, m=2)
    # ... NONLINEAR PARAMETER 
    gamma = 1.

    # -- SET AXES
    t_max, t_num = 50., 2**12
    t = np.linspace(-t_max, t_max, t_num, endpoint=False)
    w = nfft.fftfreq(t.size, d=t[1]-t[0])*2*np.pi

    # -- INITIALIZE SOLVER
    # ... SET MODEL
    model = NSE(w, -1.0, gamma)
    #model = NSE(w, beta(w), gamma)

    # ... SET Z-STEPPER
    switcher = {
        'RK2': RungeKutta2,
        'RK4': RungeKutta4
    }
    try:
        my_stepper = switcher[stepper]
    except KeyError:
        print('NOTE: STEPPER MUST BE ONE OF', list(switcher.keys()))
        raise
        exit()

    # ... SET SOLVER TYPE
    switcher = {
        'SiSSM': SiSSM(model.Lw, model.Nw, my_stepper),
        'SySSM': SySSM(model.Lw, model.Nw, my_stepper),
        'IFM': IFM_RK4IP(model.Lw, model.Nw),
        'LEM': LEM_SySSM(model.Lw, model.Nw, my_stepper),
        'MLEM': LEM_IFM(model.Lw, model.Nw)
    }
    try:
        my_solver = switcher[mode]
    except KeyError:
        print('NOTE: MODE MUST BE ONE OF', list(switcher.keys()))
        raise
        exit()

    # -- FUNCTIONS FOR ERROR ESTIMATION
    # ... AVERAGE RMS ERROR, REF. [DeVries, AIP Conference Proceedings 160, 269 (1987)]
    _RMS_error = lambda x,y: np.sqrt(np.sum(np.abs(x-y)**2)/x.size)
    # ... AVERAGE RELATIVE INTENSITY ERROR, REF. [Hult, J. Lightwave Tech., 25, 3770 (2007)]
    _RI_error = lambda x,y: np.sum(np.abs( np.abs(x)**2 - np.abs(y)**2  )/x.size/np.max(np.abs(y)**2)  )

    # -- SET TEST PULSE PROPERTIES (FUNDAMENTAL SOLITON)
    t0 = 1.                             # duration
    P0 = np.abs(beta2(0))/t0/t0/gamma   # peak-intensity
    LD = t0*t0/np.abs(beta2(0))         # dispersion length
    # ... EXACT SOLUTION
    u_exact = lambda z, t: np.sqrt(P0)*np.exp(0.5j*gamma*P0*z)/np.cosh(t/t0)
    # ... INITIAL CONDITION FOR PROPAGATION
    u0_t = u_exact(0.0, t)

    # -- SET PROPAGATION RANGE
    z_max  =   0.5*np.pi*LD             # propagate for one soliton period
    z_skip =   8                        # number of system states to skip

    data = dict()
    for z_num in [2**n for n in range(4,15)]:
        # ...  PROPAGATE INITIAL CONITION
        my_solver.set_initial_condition(w, FT(u0_t))
        my_solver.propagate(z_range = z_max, n_steps = z_num, n_skip = z_skip)

        dz =  z_max/(z_num+1)
        z_fin = my_solver.z[-1]
        u_t_fin = my_solver.utz[-1]
        u_t_fin_exact = u_exact(z_fin, t)

        # ... KEEP RESULTS
        data[dz] = (z_fin, z_num,  _RMS_error( u_t_fin, u_t_fin_exact  ),  _RI_error( u_t_fin, u_t_fin_exact  )  )

        # ... CLEAR DATA FIELDS
        my_solver.clear()

    return data


if __name__=='__main__':
    mode = sys.argv[1]
    stepper = sys.argv[2]

    data = determine_error(mode, stepper)

    #print("# (dz) (err_RMS) (err_I)")
    for dz, val in sorted(data.items()):
        z_fin, z_num, err_RMS, err_I = val
        print(dz, err_RMS, err_I )

