"""
Implements a solver base class that serves as a driver for a selection of
implemented :math:`z`-propagation algorithms.  Currently, the following
algorithms are supported:

.. autosummary::
   :nosignatures:

   SolverBaseClass
   SiSSM
   SySSM
   IFM
   LEM_SySSM
   CQE_RK4IP

A full :math:`z`-propagation scheme, i.e. a solver, is obtained by choosing one
of the implemented :math:`z`-propagation algorithms and specifying a
:math:`z`-stepping formula for the field update, see the calling structure of
the solvers below. If a user does not specify a :math:`z`-stepping formula,
each solver falls back to a reasonable default.

Further :math:`z`-propagation schemes can be implemented by using the
class :class:`SolverBaseClass`.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
from .solver_base import SolverBaseClass
from .split_step_solver import SiSSM, SySSM
from .integrating_factor_method import IFM
from .local_error_method import LEM_SySSM
from .conservation_quantity_error_method import CQE_RK4IP

# ALIAS FOR RUNGE-KUTTA IN THE INTERACTION PICTURE METHOD
IFM_RK4IP = IFM

# ALIAS FOR LOCAL ERROR METHOD
LEM = LEM_SySSM

# ALIAS FOR CONSERVATION QUANTITY ERROR METHOD
CQE = CQE_RK4IP
