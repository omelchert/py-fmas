"""
Implements several :math:`z`-propagation models based on the forward model for
the analytic signal [1,2,3,4].

.. autosummary::
   :nosignatures:

   ModelBaseClass
   FMAS_THG
   FMAS
   FMAS_S
   FMAS_S_Raman
   BMCF
   CustomModelPCF

Further :math:`z`-propagation models can be implemented by using the
class :class:`ModelBaseClass`.

References:
    [1] Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
    propagation equations for ultrashort optical pulses, Phys. Rev. E 10
    (2010) 013812, http://dx.doi.org/10.1103/PhysRevA.82.013812.

    [2] Sh. Amiranashvili, A. Demircan, Ultrashort Optical Pulse Propagation in
    terms of Analytic Signal, Adv. Opt. Tech. 2011 (2011) 989515,
    http://dx.doi.org/10.1155/2011/989515.

    [3] A. Demircan, Sh. Amiranashvili, C. Bree, C. Mahnke, F. Mitschke, G.
    Steinmeyer, Rogue wave formation by accelerated solitons at an optical
    event horizon, Appl. Phys. B 115 (2014) 343,
    http://dx.doi.org/10.1007/s00340-013-5609-9

    [4] A. Demircan, Sh. Amiranashvili, C. Bree, U. Morgner, G. Steinmeyer,
    Supercontinuum generation by multiple scatterings at a group velocity
    horizon, Opt. Exp. 22 (2014) 3866,
    https://doi.org/10.1364/OE.22.003866.

.. module:: models

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
# from .my_models import *
from .model_base import ModelBaseClass
from .fmas_thg import FMAS_THG
from .fmas import FMAS
from .fmas_s import FMAS_S
from .fmas_s_raman import FMAS_S_Raman
from .bmcf import BMCF
from .custom_model_pcf import CustomModelPCF


# ALIAS FOR FMAS_S_Raman
FMAS_S_R = FMAS_S_Raman
