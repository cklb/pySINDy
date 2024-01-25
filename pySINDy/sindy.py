"""
Derived module from sindybase.py for classical SINDy
"""
import logging
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from findiff import FinDiff
from .sindybase import SINDyBase


logger = logging.getLogger(__name__)


class SINDy(SINDyBase):

    state_var = "x"
    input_var = "u"

    """
    Sparse Identification of Nonlinear Dynamics:
    reference: http://www.pnas.org/content/pnas/113/15/3932.full.pdf
    """
    def fit(self, x_data, t_grid, poly_degree=2, cut_off=1e-3, deriv_order=2,
            u_data=None, exponents=None):
        """
        :param x_data: dynamics data to be processed
        :param t_grid: float, represents grid spacing
        :param poly_degree: degree of polynomials to be included in theta matrix
        :param cut_off: the threshold cutoff value for sparsity
        :param deriv_order: (positive) integer, derivative accuracy
        :return: a SINDy model
        """
        dt = t_grid[1:] - t_grid[:-1]
        if np.allclose(dt, dt[0]):
            args = (0, dt[0], 1)
        else:
            args = (0, t_grid, 1)

        # compute time derivative
        d_dt = FinDiff(args, acc=deriv_order)
        x_dot = d_dt(x_data)

        if x_data.ndim == 1:
            x_data = x_data[:, np.newaxis]
            x_dot = x_dot[:, np.newaxis]

        if len(x_data.shape) > 2:
            len_t = x_data.shape[-1]
            x_data = x_data.reshape((-1, len_t))
            print("The array is converted to 2D automatically: in SINDy, "
                  "each dimension except for the time (default the last dimension) "
                  "are treated equally.")
            raise NotImplementedError("reshape xdot")

        # join input if one is given
        data = self._join_args(x_data, u_data)

        # create names
        self.var_names = ["x_{}".format(i) for i in range(x_data.shape[1])]
        if u_data is not None:
            self.var_names += ["u_{}".format(i) for i in range(u_data.shape[1])]

        self._exp, self._desp, lib = self.polynomial_expansion(data,
                                                               degree=poly_degree,
                                                               exponents=exponents,
                                                               var_names=self.var_names)

        # sparse regression
        self._coef, _ = self.sparsify_dynamics(lib, x_dot, cut_off)

        # identify non trivial terms and remember them
        used_idxs = [not all(row == 0) for row in self._coef]
        self._exp = self._exp[used_idxs]
        self._desp = self._desp[used_idxs]
        self._coef = self._coef[used_idxs]

        self._deg = poly_degree

    def _join_args(self, x_data, u_data):
        if u_data is not None:
            # if u_data.ndim == 1:
            #     u_data = u_data[:, np.newaxis]
            if x_data.ndim == 2:
                if u_data.shape[0] != x_data.shape[0]:
                    u_data = u_data.T
            data = np.hstack((x_data, u_data))
        else:
            data = x_data
        return data

    def rhs(self, x, u=None):
        data = self._join_args(x, u)
        theta_mat = self.build_feature_matrix(data, self._exp)
        x_dot = theta_mat.dot(self._coef).squeeze()
        return x_dot

    def predict(self, x0, t_arr, u_arr=None):
        if self._coef is None:
            raise ValueError("Call fit first!")

        x0 = np.atleast_1d(x0)

        if u_arr is not None:
            u_callback = interp1d(t_arr,
                                  u_arr,
                                  axis=0,
                                  kind="linear",
                                  bounds_error=False,
                                  fill_value=(u_arr[0], u_arr[-1])
                                  )

        def _wrapper(t, x):
            if u_arr is not None:
                u = u_callback(t)
            else:
                u = None
            x_dot = self.rhs(x, u)
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
        return result.y.T

    def to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
