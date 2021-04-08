"""
Implements dataclass and functions for handling and reading simulation
parameters and data from and to HDF5 files.

.. autosummary::
   :nosignatures:

   SimPars
   set_par_from_dict
   fetch_par_dict_h5
   read_h5
   save_h5

.. module:: data_io

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import os
import h5py
import numpy as np
from dataclasses import dataclass


@dataclass
class SimPars:
    r"""Simulation parameter dataclass.

    Dataclass holding simulation paramters defining the computational domain,
    model, and z-propagation scheme.

    Args:
        t_max (:obj:`float`): Half-period of temporal mesh :math:`t`.
        t_num (:obj:`int`): Number of meshpoints in time-mesh :math:`t`.
        z_min (:obj:`float`): Value of first mesh-point along :math:`z`.
        z_max (:obj:`float`): Value of last mesh-point along :math:`z`.
        z_num (:obj:`int`): Number of slices along :math:`z`.
        z_skip (:obj:`int`): Number of steps to skip when accumulating data (default z_skip=1).
        beta_w (:obj:`numpy.ndarray`): Frequency-domain representation of propagation constant.
        n2 (:obj:`float`): Nonlinear refractive index.
        fR (:obj:`float`): Fractional contribution of Raman response.
        tau1 (:obj:`float`): Raman response time scale.
        tau2 (:obj:`float`): Raman response time scale.
        E_0t (:obj:`numpy.ndarray`): Real-valued optical field at :math:`z=z_{\rm{min}}`.
        out_file_path (:obj:`str`): Full path for output file.
        model_type (:obj:`str`):  Propagation model (default: FMAS_S_R).
        solver_type (:obj:`str`): Propagation algorithm (default: IFM_RK4IP).
    """
    t_max: float = None
    t_num: int = None
    z_min: float = None
    z_max: float = None
    z_num: int = None
    z_skip: int = 1
    beta_w: np.ndarray = None
    n2: float = None
    fR: float = None
    tau1: float = None
    tau2: float = None
    E_0t: np.ndarray = None
    out_file_path: str = None
    model_type: str = "FMAS_S_R"
    solver_type: str = "IFM_RK4IP"


def set_par_from_dict(sim_par, **par_dict):
    r"""Set parameters from dictionary.

    Args:
       sim_par (:obj:`dataclass`): Simulation parameter dataclass.
       **par_dict: Arbitrary keyword arguments.

    Returns:
       (:obj:`dataclass`): Simulation parameter dataclass.
    """
    opt_pars = ["model_type", "solver_type", "z_skip"]
    for attr_name in vars(sim_par).keys():
        try:
            attr_value = par_dict[attr_name]
            setattr(sim_par, attr_name, attr_value)
        except KeyError:
            if attr_name in opt_pars:
                pass
            else:
                print(f"Attribute {attr_name} is not available in dict!")
                raise
    return sim_par


def fetch_par_dict_h5(file_path):
    r"""Fetch parameter dictionary from file.

    Args:
        file_path (:obj:`str`): Location of HDF5 input file.

    Returns:
        :obj:`dict`: Dictionary holding parameters from file.
    """
    data = dict()
    with h5py.File(file_path, "r") as f:
        for par_name in f.keys():
            par_value = np.array(f.get(par_name), dtype=f.get(par_name).dtype)
            data[par_name] = par_value
    return data


def fetch_parameters_from_file_h5(file_path):
    r"""Fetch parameters from file.

    Args:
        file_path (:obj:`str`): Location of HDF5 input file.

    Returns:
       (:obj:`dataclass`): Simulation parameter dataclass.
    """
    sim_par = SimPars()
    par_dict = fetch_par_dict_h5(file_path)
    sim_par = set_par_from_dict(sim_par, **par_dict)
    return sim_par


# ALIAS FOR fetch_parameters_from_file_h5
read_h5 = fetch_parameters_from_file_h5


def save_data_to_file_h5(out_path, **results):
    r"""Save data in HDF5 format.

    Args:
       out_path (:obj:`str`): Name for ouput file.
       **results: Arbitrary keyword arguments.
    """
    dir_name = os.path.dirname(out_path)
    file_basename = os.path.basename(out_path)
    file_extension = file_basename.split(".")[-1]

    try:
        os.makedirs(dir_name)
    except OSError:
        pass

    with h5py.File(out_path, "w") as f:
        for key, val in results.items():
            f.create_dataset(key, data=val)


# ALIAS FOR save_data_to_file_h5
save_h5 = save_data_to_file_h5
