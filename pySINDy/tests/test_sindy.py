# pylint: skip-file

import warnings
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from ..sindy import SINDy

class TestSINDy(TestCase):
    def test_shape1(self):
        warnings.simplefilter("ignore")
        model = SINDy()
        data = np.random.rand(10,)
        dt = 0.1
        model.fit(data, dt)
        assert True


class TestSINDyC(TestCase):

    def setUp(self):

        self.u_func = np.sin

        def rhs(t,y):
            u = self.u_func(t)
            y_dt = np.array([y[1], -y[0] + u])
            return y_dt

        t0 = 0
        te = 100
        t_grid = np.linspace(t0, te, 1000)
        y0 = np.array([1, 0])
        res = solve_ivp(rhs, (t0, te), y0, t_eval=t_grid)

        self.t_data = t_grid
        self.x_data = res.y.T
        self.u_data = self.u_func(t_grid)[:, None]

    def test_run(self):
        # fit
        model = SINDy()

        exponents = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        model.fit(self.x_data, self.t_data,
                  poly_degree=5,
                  exponents=exponents,
                  u_data=self.u_data)
        model.plot_coefficients()
        plt.show()

        # predict
        res = model.predict(self.x_data[0], self.t_data, self.u_data)
        for meas, pred in zip(self.x_data.T, res.T):
            l = plt.plot(self.t_data, pred, label="pred")
            plt.plot(self.t_data, meas, "--")

        plt.show()

    def test_shape1(self):
        warnings.simplefilter("ignore")
        model = SINDy()
        data = np.random.rand(10,)
        dt = 0.1
        model.fit(data, dt)
        assert True
