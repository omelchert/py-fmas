"""
Script demonstrating how the convenience class PropConst can be used to analyze
a propagation constant.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from fmas.propagation_constant import PropConst


def get_beta_fun_ESM():
    p = np.poly1d(
        (16.89475, 0.0, -319.13216, 0.0, 34.82210, 0.0, -0.992495, 0.0, 0.0010671)[::-1]
    )
    q = np.poly1d(
        (1.00000, 0.0, -702.70157, 0.0, 78.28249, 0.0, -2.337086, 0.0, 0.0062267)[::-1]
    )
    c = 0.29979  # (micron/fs)
    return lambda w: (1 + p(w) / q(w)) * w / c


# -- WRAP PROPAATION CONSTANT
beta_fun = get_beta_fun_ESM()
pc = PropConst(beta_fun)

# -- FIND ZERO DISPERSION POINT
w_Z = pc.find_root_beta2(1.3, 2.2)
print("w_Z = %lf rad/fs" % (w_Z))

# -- FIND GV MATCHED PARTNER FREQUENCY
w_S = 1.5  # (rad/fs)
w_GVM = pc.find_match_beta1(w_S, w_Z, 2.5)
print("w_GVM = %lf rad/fs " % (w_GVM))

# -- GV MISMATCH FOR S AND DW1
w_DW1 = 2.06  # (rad/fs)
dvg = pc.vg(w_DW1) - pc.vg(w_S)
print("dvg = %lf micron/fs" % (dvg))

# -- LOCAL EXPANSION COEFFICIENTS
betas = pc.local_coeffs(w_S, n_max=4)
for n, bn in enumerate(betas):
    print("b_%d = %lf fs^%d/micron" % (n, bn, n))
