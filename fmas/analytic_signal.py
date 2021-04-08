"""
Implements class `AnalyticSignal`, which uses the frequncy-domain algorithm
detailed in [M1999]_ to compute the complex-valued analytic signal for a
real-valued discrete-time field.

.. [M1999] S. L. Marple,
   Computing the Discrete-Time Analytic signal via FFT,
   IEEE Transactions on Signal Processing, 47 (1999) 2600.

.. module:: analytic_signal

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .config import FTFREQ, FT, IFT


class AnalyticSignal:
    r"""Class converting real-valued field to analytic signal.

    Implements methods that take a real-valued :math:`N`-point discrete-time
    field :math:`E` and provide access to the complex-valued discrete-time
    analytic signal :math:`\mathcal{E}`, both in time-domain and
    Fourier-domain.

    Note:
        The class implements the frequency-domain algorithm specified in
        [M1999]_, given by the three-step procedure

        1. Compute :math:`E_\omega`, i.e. the :math:`N`-point discrete
        Fourier-trasform (DFT) of the real-valued discrete-time field
        :math:`E`.

        2. Compute the :math:`N`-point discrete-time analytic signal in the
        Fourier-domain, given by

            .. math:: \mathcal{E}_\omega[m] = \begin{cases}
                            E_\omega[0],    & m=0,\\
                            2E_\omega[m],   & 1\leq m \leq N/2-1,\\
                            E_\omega[N/2],  & m = N/2,\\
                            0,              & N/2+1 \leq m \leq N - 1.
                      \end{cases}

        3. Compute the complex-valued :math:`N`-point discrete-time analytic
        signal :math:`\mathcal{E}` using an inverse DFT.

    Args:
        x (:obj:`numpy.ndarray`): Real-valued field :math:`E`.

    Attributes:
        x (:obj:`numpy.ndarray`): Real-valued field :math:`E`.
        num (:obj:`int`): Number :math:`N` of field points.

    """

    def __init__(self, x):
        self.x = np.asarray(x)
        self.num = self.x.size

    @property
    def w_rep(self):
        r""":obj:`numpy.ndarray`: Frequency-domain representation :math:`\mathcal{E}_\omega` of the analytic signal."""
        num, x = self.num, self.x
        tmp = FT(x)
        tmp[1 : int(num / 2)] = 2 * tmp[1 : int(num / 2)]
        tmp[int(num / 2) + 1] = tmp[int(num / 2) + 1]
        tmp[int(num / 2) + 1 :] = 0j
        return tmp

    @property
    def t_rep(self):
        r""":obj:`numpy.ndarray`: Time-domain representation :math:`\mathcal{E}` of the analytic signal."""
        return IFT(self.w_rep)

    def test_recover_original_field(self):
        r"""Check if real part of analytic signal equals original field.

        Real part of the discrete-time analytic signal must be equal to the original field, i.e.

        .. math:: \mathsf{Re}(\mathcal{E}[n]) = E[n], \quad 0\leq n\leq N-1,

        see Eq. (6) of [M1999]_.
        """
        z = self.t_rep
        np.testing.assert_allclose(self.x, np.real(z), rtol=1e-8)

    def test_orthogonality(self):
        r"""Check if real and imaginary parts are orthogonal.

        Real and imaginary parts of the analytic signal must be orthogonal, i.e.

        .. math:: \sum_{n=0}^{N-1} \mathsf{Re}(\mathcal{E}[n])\,\mathsf{Im}(\mathcal{E}[n]) = 0,

        see Eq. (7) of [M1999]_.
        """
        z = self.t_rep
        z_r, z_im = np.real(z), np.imag(z)
        assert np.sum(z_r * z_im) < 1e-10


AS = AnalyticSignal

if __name__ == "__main__":
    x = [4, 2, -2, -1, 3, 1, -3, 1]
    Z = AnalyticSignal(x)
    Z.test_orthogonality()
    Z.test_recover_original_field()
