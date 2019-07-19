# pylint: skip-file

import warnings
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

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
        te = 10
        t_grid = np.linspace(t0, te, 100)
        y0 = np.array([1, 0])
        res = solve_ivp(rhs, (t0, te), y0, t_grid=t_grid)

        self.t_data = t_grid
        self.y_data = res.y.T
        self.u_data = self.u_func(t_grid)

    def test_fit(self):
        model = SINDy()
        model.fit(self.y_data, self.t_data,
                  poly_degree=1, input_data=self.u_data)
        print(model._coef)

    def test_shape1(self):
        warnings.simplefilter("ignore")
        model = SINDy()
        data = np.random.rand(10,)
        dt = 0.1
        model.fit(data, dt)
        assert True
