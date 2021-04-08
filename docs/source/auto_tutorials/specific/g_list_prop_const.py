r"""
Implemented propagation constants
=================================

This example shows the group-velocity and group-velocity dispersion profiles of
several propagation constants implemnted in module `propagation_constant`.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
###############################################################################
# We first start by importing the functionality of numpy and fmas into the
# current namespace.  Specifically, we import the module `propagation_constant`
# that contains the convenience class `PropConst` as well as the implemented
# propagation constant. A function that allows to plot the group-velocity  and
# group-velocity dispersion profiles is imported from module `tools`.

import numpy as np
import fmas
import fmas.propagation_constant as prop_const
from fmas.tools import plot_details_prop_const

###############################################################################
# Below we show a list of the basic properties of several propatgation
# constants implemented in `fmas`.
#
# NLPM750
# -------

pc = prop_const.PropConst(prop_const.define_beta_fun_NLPM750())
w  = np.linspace(1.3, 3.2, 200)
plot_details_prop_const(w, pc.vg(w), pc.beta2(w))

###############################################################################
# with zero-dispersion points

w_Z1 = pc.find_root_beta2(1.4, 1.7)
w_Z2 = pc.find_root_beta2(2.2, 2.5)

print('w_Z1 = ', w_Z1)
print('w_Z2 = ', w_Z2)

###############################################################################
# ESM 
# ---

pc = prop_const.PropConst(prop_const.define_beta_fun_ESM())
w  = np.linspace(1., 3.5, 200)
plot_details_prop_const(w, pc.vg(w), pc.beta2(w))

# sphinx_gallery_thumbnail_number = 2

###############################################################################
# with zero-dispersion point

w_Z1 = pc.find_root_beta2(1.5, 2.0)

print('w_Z1 = ', w_Z1)


###############################################################################
# ZBLAN
# -----

pc = prop_const.PropConst(prop_const.define_beta_fun_ZBLAN())
w  = np.linspace(0.5, 5., 200)
plot_details_prop_const(w, pc.vg(w), pc.beta2(w))

###############################################################################
# with zero-dispersion point


w_Z1 = pc.find_root_beta2(1., 1.3)

print('w_Z1 = ', w_Z1)


###############################################################################
# Silicon slot waveguide 
# ----------------------

pc = prop_const.PropConst(prop_const.define_beta_fun_slot_waveguide_Zhang2012())
w  = np.linspace(0.86, 1.37, 200)
plot_details_prop_const(w, pc.vg(w), pc.beta2(w))

###############################################################################
# with four zero-dispersion points

w_Z1 = pc.find_root_beta2(0.9, 0.95)
w_Z2 = pc.find_root_beta2(0.95, 1.05)
w_Z3 = pc.find_root_beta2(1.1, 1.2)
w_Z4 = pc.find_root_beta2(1.25, 1.3)

print('w_Z1 = ', w_Z1)
print('w_Z2 = ', w_Z2)
print('w_Z3 = ', w_Z3)
print('w_Z4 = ', w_Z4)
