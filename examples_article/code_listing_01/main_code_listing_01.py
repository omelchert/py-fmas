"""
Exemplary workflow using the py-fmas library code.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import fmas
import numpy as np

glob = fmas.data_io.read_h5("in_file.h5")

grid = fmas.grid.Grid(
    t_max=glob.t_max, t_num=glob.t_num, z_max=glob.z_max, z_num=glob.z_num
)

model = fmas.models.FMAS_S_Raman(
    w=grid.w, beta_w=glob.beta_w, n2=glob.n2, fR=glob.fR, tau1=glob.tau1, tau2=glob.tau2
)

ic = fmas.analytic_signal.AS(glob.E_0t)


def Cp(i, zi, w, uw):
    Iw = np.abs(uw) ** 2
    return np.sum(Iw[w > 0] / w[w > 0])


solver = fmas.solver.IFM_RK4IP(model.Lw, model.Nw, user_action=Cp)
solver.set_initial_condition(grid.w, ic.w_rep)
solver.propagate(z_range=glob.z_max, n_steps=glob.z_num, n_skip=glob.z_skip)

res = {"t": grid.t, "z": solver.z, "w": solver.w, "u": solver.utz, "Cp": solver.ua_vals}
fmas.data_io.save_h5("out_file.h5", **res)

fmas.tools.plot_evolution(
    solver.z, grid.t, solver.utz, t_lim=(-500, 2200), w_lim=(1.0, 4.0)
)
