import sys, os
import time
import numpy as np
import logging as log
from .config import FTFREQ, FT, IFT
from .solver import IFM_RK4IP, SiSSM, SySSM, LEM, CQE
from .models import FMAS_S, FMAS_S_Raman
from .data_io import read_h5, save_h5
from .grid import Grid
from .analytic_signal import AnalyticSignal


def run(file_name, model_type = 'FMAS_S_R', solver_type = 'IFM_RK4IP'):

    glob = read_h5(file_name)

    grid = Grid(t_max=glob.t_max, t_num=glob.t_num, z_max=glob.z_max, z_num=glob.z_num)

    # -- SET MODEL TYPE
    model_switch = {
        "FMAS_S": FMAS_S(w=grid.w, beta_w=glob.beta_w, n2=glob.n2),
        "FMAS_S_R": FMAS_S_Raman(
            w=grid.w,
            beta_w=glob.beta_w,
            n2=glob.n2,
            fR=glob.fR,
            tau1=glob.tau1,
            tau2=glob.tau2,
        ),
    }
    try:
        model = model_switch[model_type]
    except KeyError:
        print("MODEL_TYPE MUST BE ONE OF", list(model_switch.keys()))
        raise
        exit()

    ic = AnalyticSignal(glob.E_0t)

    # -- SET SOLVER TYPE
    solver_switch = {
        "SiSSM": SiSSM,
        "SySSM": SySSM,
        "IFM_RK4IP": IFM_RK4IP,
        "LEM": LEM,
        "CQE": CQE
    }
    try:
        Solver = solver_switch[solver_type]
    except KeyError:
        print("SOLVER_TYPE MUST BE ONE OF", list(solver_switch.keys()))
        raise
        exit()

    solver = Solver(model.Lw, model.Nw, user_action=model.claw)

    solver.set_initial_condition(grid.w, ic.w_rep)

    solver.propagate(z_range=glob.z_max, n_steps=glob.z_num, n_skip=glob.z_skip)

    res = {
        "t": grid.t,
        "z": solver.z,
        "w": solver.w,
        "u": solver.utz,
        "Cp": solver.ua_vals,
    }

    return res


if __name__ == "__main__":
    file_name = sys.argv[1]
    res = run(file_name)
    save_h5("out_file.h5", **res)
