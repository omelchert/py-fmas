"""
The :math:`z`-propagation schemes implemented with the pyFMAS package are
specified by a :math:`z`-propagation algorithm along with a :math:`z`-stepping
formula that performs the field update for a single step.
Implements are the subsequent :math:`z`-stepping formulas.

.. autosummary::
   :nosignatures:

   RungeKutta2
   RungeKutta4

.. [NR1992] W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
    Numerical Recipes in C: The art of scientific computing (Chapter 16.1),
    Cambridge University Press (1992).

.. module:: stepper

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""


def RungeKutta2(fun, z, uw, dz):
    r"""Second-order Runge-Kutta formula

    Implements second-order Runge-Kutta formula for :math:`z`-stepping
    [NR1992]_.  Achieves local error :math:`\mathcal O(\Delta z^3)`, with step
    size :math:`\Delta z`.

    Args:
        fun (:obj:`function`):
            Function evaluating the evolution rate of the ODE.
        z (:obj:`float`):
            Current :math:`z`-value.
        uw (:obj:`numpy.ndarray`, 1-dim):
            Frequency-domain representation of the current field.
        dz (:obj:`float`):
            Step size.
    """
    k1 = fun(z, uw)
    k2 = fun(z + 0.5 * dz, uw + 0.5 * dz * k1)
    return uw + dz * k2


def RungeKutta4(fun, z, uw, dz):
    r"""Fourth-order Runge-Kutta formula

    Implements fourth-order Runge-Kutta formula for :math:`z`-stepping
    [NR1992]_.  Achieves local error :math:`\mathcal O(\Delta z^5)`, with step
    size :math:`\Delta z`.

    Args:
        fun (:obj:`function`):
            Function evaluating the evolution rate of the ODE.
        z (:obj:`float`):
            Current :math:`z`-value.
        uw (:obj:`numpy.ndarray`, 1-dim):
            Frequency-domain representation of the current field.
        dz (:obj:`float`):
            Step size.
    """
    k1 = fun(z, uw)
    k2 = fun(z + 0.5 * dz, uw + 0.5 * dz * k1)
    k3 = fun(z + 0.5 * dz, uw + 0.5 * dz * k2)
    k4 = fun(z + dz, uw + dz * k3)
    return uw + dz * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# EOF: z_stepper.py
