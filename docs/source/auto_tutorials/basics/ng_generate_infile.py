r"""
Generating an input file
========================

This examples shows how to generate an input file in HDF5-format, which can
then be processed by the `py-fmas` library code.

This is useful when the project-specific code is separate from the `py-fmas`
library code.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>

"""

###############################################################################
# We start by importing the required `py-fmas` functionality. Since the
# file-input for `py-fmas` is required to be provided in HDF5-format, we need
# some python package that offers the possibility to read and write this
# format.  Here we opted for the python module h5py which is listed as one of
# the dependencies of the `py-fmas` package.
import h5py
import numpy as np
import numpy.fft as nfft

###############################################################################
# We then define the desired propagation constant 

def beta_fun_detuning(w):
    r'''Function defining propagation constant

    Implements group-velocity dispersion with expansion coefficients
    listed in Tab. I of Ref. [1]. Expansion coefficients are valid for
    :math:`lambda = 835\,\mathrm{nm}`, i.e. for :math:`\omega_0 \approx
    2.56\,\mathrm{rad/fs}`.

    References:
        [1] J. M. Dudley, G. Genty, S. Coen,
        Supercontinuum generation in photonic crystal fiber,
        Rev. Mod. Phys. 78 (2006) 1135,
        http://dx.doi.org/10.1103/RevModPhys.78.1135

    Note:
        A corresponding propagation constant is implemented as function
        `define_beta_fun_PCF_Ranka2000` in `py-fmas` module
        `propatation_constant`.

    Args:
        w (:obj:`numpy.ndarray`): Angular frequency detuning.

    Returns:
        :obj:`numpy.ndarray` Propagation constant as function of
        frequency detuning.
    '''
    # ... EXPANSION COEFFICIENTS DISPERSION
    b2 = -1.1830e-2     # (fs^2/micron)
    b3 = 8.1038e-2      # (fs^3/micron)
    b4 = -0.95205e-1    # (fs^4/micron)
    b5 = 2.0737e-1      # (fs^5/micron)
    b6 = -5.3943e-1     # (fs^6/micron)
    b7 = 1.3486         # (fs^7/micron)
    b8 = -2.5495        # (fs^8/micron)
    b9 = 3.0524         # (fs^9/micron)
    b10 = -1.7140       # (fs^10/micron)
    # ... PROPAGATION CONSTANT (DEPENDING ON DETUNING)
    beta_fun_detuning = np.poly1d([b10/3628800, b9/362880, b8/40320,
        b7/5040, b6/720, b5/120, b4/24, b3/6, b2/2, 0., 0.])
    return beta_fun_detuning(w)

###############################################################################
# Next, we define all parameters needed to specify a simulation run 

# -- DEFINE SIMULATION PARAMETERS
# ... COMPUTATIONAL DOMAIN 
t_max = 3500.       # (fs)
t_num = 2**14       # (-)
z_max = 0.1*1e6     # (micron)
z_num = 4000        # (-)
z_skip = 20         # (-)
t = np.linspace(-t_max, t_max, t_num, endpoint=False)
w = nfft.fftfreq(t.size, d=t[1]-t[0])*2*np.pi
# ... MODEL SPECIFIC PARAMETERS 
# ... PROPAGATION CONSTANT
c = 0.29979         # (fs/micron)
lam0 = 0.835        # (micron)
w0 = 2*np.pi*c/lam0 # (rad/fs)
beta_w = beta_fun_detuning(w-w0)
gam0 = 0.11e-6      # (1/W/micron)
n2 = gam0*c/w0      # (micron^2/W)
# ... PARAMETERS FOR RAMAN RESPONSE 
fR = 0.18           # (-)
tau1= 12.2          # (fs)
tau2= 32.0          # (fs)
# ... INITIAL CONDITION
t0 = 28.4           # (fs)
P0 = 1e4            # (W)
E_0t_fun = lambda t: np.real(np.sqrt(P0)/np.cosh(t/t0)*np.exp(-1j*w0*t))
E_0t = E_0t_fun(t)

###############################################################################
# The subsequent code will store the simulation parameters defined above to the
# file `input_file.h5` in the current working directory.

def save_data_hdf5(file_path, data_dict):
    with h5py.File(file_path, 'w') as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=val)

data_dict = {
   't_max': t_max,
   't_num': t_num,
   'z_min': 0.0,
   'z_max': z_max,
   'z_num': z_num,
   'z_skip': z_skip,
   'E_0t': E_0t,
   'beta_w': beta_w,
   'n2': n2,
   'fR': fR,
   'tau1': tau1,
   'tau2': tau2,
   'out_file_path': 'out_file.h5'
}

save_data_hdf5('input_file.h5', data_dict)


###############################################################################
# An example, showing how to use `py-fmas` as a black-box simulation tool that
# performs a simulation run for the propagation scenario stored under the file
# `input_file.h5` is available under the link below:
#
# :ref:`sphx_glr_auto_tutorials_basics_g_app.py`

