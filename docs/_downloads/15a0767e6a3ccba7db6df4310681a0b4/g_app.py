r"""
Using `fmas` as a black-box application
=======================================

This examples shows how to use `py-fmas` as a black-box application, that
only requires a minimal amount of scripting.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""


###############################################################################
# We start by simply importing the required `fmas` into the current namespace.
#

import fmas

###############################################################################
# If an adequate input file is located within the current working directory,
# `fmas` can be used as shown below. It features a particular function called
# `run`, which reads-in the propagation setting stored in the input file
# `input_file.h5` and runs the simulaton

res = fmas.run('input_file.h5', model_type='FMAS_S_R', solver_type='IFM_RK4IP')

###############################################################################
# An example that shows how an adequate input file can be generated via python
# is shown under the link below:
#
# :ref:`sphx_glr_auto_tutorials_basics_ng_generate_infile.py`
#
# After the proapgation algorithm (specified in `input_file.h5`) terminates,
# a simple dictionary data structure with the following keys is available

print(res.keys())

###############################################################################
# A simple plot that shows the result of the simulation run can be produced
# using function `plot_evolution` implemented in module `tools`

from fmas.tools import plot_evolution
plot_evolution( res['z'], res['t'], res['u'], t_lim=(-500,2200), w_lim=(1.,4.))

###############################################################################
# The results can be stored for later postprocessing using the function
# `save_h5` implemented in module `data_io`. It will generate a file
# `out_file.h5` with HDF5 format in the current working directory

from fmas.data_io import save_h5
save_h5('out_file.h5', **res)
