r"""
Extending `py-fmas` by `optfrog` spectrograms
=============================================

This examples shows how to use the `py-fmas` library code in conjunction
with the optFrog spectrogram tool.

In particular, the example details a numerical experiment performing pulse
propagation in terms of the simplified forward model for the analytic signal
including the Raman effect [1]. Here, the model is used to perform simulations
on supercontinuum generation for an instance of a highly nonlinear photonic
crystal fiber (PCF) with an anomalous group-velocity dispersion regime [2].
The example also includes data postprocessing by calculating an analytic signal
spectrum with optimized time-frequency resolution using the `optfrog`-tool [3].

An example that shows how to use the simple `py-fmas` native spectrogram is
shown under the link below:

:ref:`sphx_glr_auto_tutorials_specific_g_spectrogram.py`

Note:
    * For this exampe to run, the optfrog tool needs to be installed [3].
    * The `py-fmas` package includes a simple spectrogram in module `tools`.
      The `optfrog` Python package however includes extended functionality by
      allowing a user to calculate spectrograms with optimized time and
      frequency resolution.

References:
    [1] A. Demircan, Sh. Amiranashvili, C. Bree, C. Mahnke, F. Mitschke, G.
    Steinmeyer, Rogue wave formation by accelerated solitons at an optical
    event horizon, Appl. Phys. B 115 (2014) 343,
    http://dx.doi.org/10.1007/s00340-013-5609-9

    [2] J. M. Dudley, G. Genty, S. Coen,
    Supercontinuum generation in photonic crystal fiber,
    Rev. Mod. Phys. 78 (2006) 1135,
    http://dx.doi.org/10.1103/RevModPhys.78.1135

    [3] O. Melchert, B. Roth, U. Morgner, A. Demircan,
    OptFROG — Analytic signal spectrograms with optimized time–frequency resolution,
    SoftwareX 10 (2019) 100275,
    https://doi.org/10.1016/j.softx.2019.100275,
    code repository: https://github.com/ElsevierSoftwareX/SOFTX_2019_130.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import fmas
import numpy as np
from fmas.models import CustomModelPCF
from fmas.solver import IFM_RK4IP
from fmas.analytic_signal import AS
from fmas.grid import Grid
from fmas.tools import plot_spectrogram
from optfrog import optFrog


def main():
    # -- DEFINE SIMULATION PARAMETERS
    # ... COMPUTATIONAL DOMAIN
    t_max = 3500.       # (fs)
    t_num = 2**14       # (-)
    z_max = 0.10*1e6    # (micron)
    z_num = 8000        # (-)
    z_skip = 10         # (-)
    # ... INITIAL CONDITION
    P0 = 1e4            # (W)
    t0 = 28.4           # (fs)
    w0 = 2.2559         # (rad/fs)
    E_0t_fun = lambda t: np.real(np.sqrt(P0)/np.cosh(t/t0)*np.exp(-1j*w0*t))

    # -- INITIALIZATION STAGE
    grid = Grid( t_max = t_max, t_num = t_num, z_max = z_max, z_num = z_num)
    model = CustomModelPCF(w=grid.w)
    solver = IFM_RK4IP( model.Lw, model.Nw, user_action = model.claw)
    solver.set_initial_condition( grid.w, AS(E_0t_fun(grid.t)).w_rep)
    # -- RUN SIMULATION
    solver.propagate( z_range = z_max, n_steps = z_num, n_skip = z_skip)

    # -- POSTPRICESSING: COMPUTE SPECTROGRAM USING OPTFROG 
    # ... Z-DISTANCE, Z-INDEX AND FIELD FOR WHICH TO COMPUTE TRACE
    z0 = 0.08e6  # (micron)
    z0_idx = np.argmin(np.abs(solver.z-z0))
    Et = solver.utz[z0_idx]
    # ... WINDOW FUNCTION FOR SIGNAL LOCALIZATION
    def window_fun(s0):
        return lambda t: np.exp(-t**2/2/s0/s0)/np.sqrt(2.*np.pi)/s0
    # ... OPTFROG TRACE
    res = optFrog(
        grid.t,                         # TEMPORAL GRID
        Et,                             # ANALYTIC SIGNAL 
        window_fun,                     # WINDOW FUNCTION
        tLim = (-500.0, 3200.0, 10),    # (tmin, fs) (tmax, fs) (nskip)
        wLim = (0.9, 4.1, 3)            # (wmin, fs) (wmax, fs) (nskip)
    )
    # ... SHOW SPECTROGRAM
    plot_spectrogram(res.tau, res.w, res.P)


if __name__=='__main__':
    main()
