"""
Module containing grid data structure specifying the computational domain.

.. module:: grid

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from .config import FTFREQ, FT, IFT, W_MAX_FAC


class Grid:
    r"""Data structure specifying the discretized computational domain.

    The computational domain is discretized by using a time mesh and
    z-coordinate mesh with uniform mesh-widths.

    Args:
        t_max (float): temporal mesh extends from -t_max to t_max.
        t_num (int): number of meshpoints for temporal mesh.
        z_max (float): propagation range.
        z_int (int): number of z-steps.

    Attributes:
        t_max (float): time mesh extending from -t_max to t_max.
        t_num (int): number of time mesh mesh-points.
        z_max (float): propagation range.
        z_int (int): number of z-steps.
        t (np.ndarray): array representing time mesh.
        dt (float): time mesh-width.
        z (np.ndarray): z-coordinate mesh.
        dz (float): z-coordinate mesh width.
    """

    def __init__(self, t_max, t_num, z_max=None, z_num=None):
        self.t_max = t_max
        self.t_num = t_num
        self.z_max = z_max
        self.z_num = z_num
        self.t, self.dt = self._set_t_axis()
        self.w, self.dw = self._set_w_axis()
        if z_max is not None:
            self.z, self.dz = self._set_z_axis()

    def _set_t_axis(self):
        r"""Set temporal mesh.

        Returns:
            t (np.ndarray): temporal mesh.
        """
        return np.linspace(
            -self.t_max, self.t_max, self.t_num, endpoint=False, retstep=True
        )

    def _set_w_axis(self):
        r"""Set angular frequency mesh.

        Returns:
            w (np.ndarray): angular frequency mesh.
        """
        w = FTFREQ(self.t.size, d=self.dt) * 2 * np.pi
        dw = w[1] - w[0]
        return w, dw

    def _set_z_axis(self):
        r"""Set z-mesh.

        Returns:
            z (np.ndarray): z-mesh.
        """
        return np.linspace(0, self.z_max, self.z_num + 1, retstep=True)


if __name__ == "__main__":
    grid = Grid(3500, 1000, 30, 300)
    print(grid.dt)
