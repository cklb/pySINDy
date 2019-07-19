"""
Derived module from sindybase.py for classical SINDy
"""
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from findiff import FinDiff
from .sindybase import SINDyBase


logger = logging.getLogger(__name__)


class SINDy(SINDyBase):
    """
    Sparse Identification of Nonlinear Dynamics:
    reference: http://www.pnas.org/content/pnas/113/15/3932.full.pdf
    """
    def fit(self, data, t_grid, poly_degree=2, cut_off=1e-3, deriv_order=2,
            input_data=None):
        """
        :param data: dynamics data to be processed
        :param t_grid: float, represents grid spacing
        :param poly_degree: degree of polynomials to be included in theta matrix
        :param cut_off: the threshold cutoff value for sparsity
        :param deriv_order: (positive) integer, derivative accuracy
        :return: a SINDy model
        """

        if len(data.shape) == 1:
            data = data[np.newaxis, ]

        len_t = data.shape[-1]

        if len(data.shape) > 2:
            data = data.reshape((-1, len_t))
            print("The array is converted to 2D automatically: in SINDy, "
                  "each dimension except for the time (default the last dimension) "
                  "are treated equally.")
        dt = t_grid[1:] - t_grid[:-1]
        if np.allclose(dt, dt[0]):
            args = (0, dt[0], 1)
        else:
            args = (0, t_grid, 1)

        # compute time derivative
        d_dt = FinDiff(args, acc=deriv_order)
        x_dot = d_dt(data)

        # prepare for the library
        lib, self._desp = self.polynomial_expansion(data, degree=poly_degree)

        # sparse regression
        self._coef, _ = self.sparsify_dynamics(lib, x_dot, cut_off)

        # identify non trivial terms and remember them
        self._deg = poly_degree

        return self

    def rhs(self, q_data):
        theta_mat, self._desp = self.polynomial_expansion(q_data,
                                                          degree=self._deg)
        x_dot = theta_mat.dot(self._coef)
        return x_dot

    def predict(self, x0, u_arr, t_arr):
        u_callback = interp1d(t_arr,
                              u_arr,
                              axis=1,
                              kind="linear",
                              bounds_error=False,
                              fill_value=(u_arr[..., 0], u_arr[..., -1])
                              )

        def _wrapper(t, x):
            u = u_callback(t)
            q = np.atleast_2d(np.hstack((x, u)))
            x_dot = self.rhs(q)
            return x_dot.squeeze()

        np.seterr(under="warn")
        result = solve_ivp(_wrapper,
                           t_span=(t_arr[0], t_arr[-1]),
                           t_eval=t_arr,
                           y0=x0,
                           rtol=1e-6,
                           atol=1e-6,
                           )
        if not result.success:
            logger.error(result.message)
        return result.y
