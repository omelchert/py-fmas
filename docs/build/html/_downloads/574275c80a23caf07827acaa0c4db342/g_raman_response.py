r"""
Using a specific Raman response function
========================================

This examples shows how one of the more specific Raman response functions,
implemented in modeule `raman_response`, can be used with the class
:class:`FMAS_S_Raman`.

In particular, this example shows how a Lin-Agrawal type Raman response model
can be used with the above :math:`z`-propagation model.

References:
    [1] Q. Lin, G. P. Agrawal, Raman response function for silica fibers,
    Optics Letters, 31 (2006) 3086, https://doi.org/10.1364/JOSAB.6.001159.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import fmas
import numpy as np
from fmas.models import CustomModelPCF
from fmas.config import FTFREQ, FT, IFT
from fmas.solver import IFM_RK4IP
from fmas.analytic_signal import AS
from fmas.grid import Grid
from fmas.tools import plot_evolution

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
# ... COMPUTATIONAL DOMAIN
grid = Grid( t_max = t_max, t_num = t_num, z_max = z_max, z_num = z_num)
# ... CUSTOM PROPAGATION MODEL
model = CustomModelPCF(w=grid.w)

###############################################################################
# To use a Raman response function different from the default, we simply import
# the desired model from the module `raman_response`.  We here opt to use a
# Lin-Agrawal type Raman response, which is implemented as function `h_LA`. The
# function can be used to simply overwrite the default response function as
# shown below. If a different value for the fractional Raman contribution is
# needed, the corresponding class attribute can be overwritten as well:

# -- Imports Lin-Agrawal type response function
from fmas.raman_response import h_LA

model.hRw = h_LA(grid.t) # overwrite default resonse function
model.fR = 0.18          # overwrite default fractional Raman response if needed


# ... PROPAGATION ALGORITHM 
solver = IFM_RK4IP( model.Lw, model.Nw, user_action = model.claw)
solver.set_initial_condition( grid.w, AS(E_0t_fun(grid.t)).w_rep)

# -- RUN SIMULATION
solver.propagate( z_range = z_max, n_steps = z_num, n_skip = z_skip)

# -- SHOW RESULTS
plot_evolution( solver.z, grid.t, solver.utz,
    t_lim = (-500,2200), w_lim = (1.,4.), DO_T_LOG = False)

