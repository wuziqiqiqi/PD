import unittest
from clease import PhysicalRidge
from clease.physical_ridge import random_cv_hyper_opt
import numpy as np


class TestPhysicalRidge(unittest.TestCase):
    def test_size_from_name(self):
        phys_ridge = PhysicalRidge()
        names = ['c0', 'c1_1', 'c2_d0000_0_00', 'c3_d1223_0_11']
        phys_ridge.sizes_from_names(names)
        expect = [0, 1, 2, 3]
        self.assertListEqual(expect, phys_ridge.sizes)

    def test_dia_from_name(self):
        phys_ridge = PhysicalRidge()
        names = ['c0', 'c1_1', 'c2_d0000_0_00', 'c3_d1223_0_11']
        phys_ridge.diameters_from_names(names)
        expect = [0, 0, 0, 1223]
        self.assertListEqual(phys_ridge.diameters, expect)

    def test_fit(self):
        phys_ridge = PhysicalRidge()
        names = ['c0', 'c1_1', 'c2_d0000_0_00', 'c3_d0002_0_11']

        X = np.random.rand(10, 4)
        X[:, 0] = 1.0
        y = np.random.rand(10)

        with self.assertRaises(ValueError):
            phys_ridge.fit(X, y)

        phys_ridge.sizes_from_names(names)
        phys_ridge.diameters_from_names(names)
        phys_ridge.fit(X, y)

        # Confirm that hyper optimization is working
        params = {
            'lamb_dia': [1.0, 2.0, 3.0, 4.0],
            'lamb_size': [1.0, 2.0, 3.0],
            'dia_decay': ['linear', 'exponential'],
            'size_decay': ['linear', 'exponential']
        }

        random_cv_hyper_opt(phys_ridge, params, X, y, cv=5, num_trials=5)

    def test_constraints(self):
        phys_ridge = PhysicalRidge(lamb_dia=0.0, lamb_size=1e-4,
                                   normalize=False)
        X = np.zeros((3, 5))
        x = np.array([0.0, 2.0, 4.0])
        for i in range(5):
            X[:, i] = x**i
        y = 2.0*x + x**2
        phys_ridge.diameters = np.zeros(5)
        phys_ridge.sizes = 2*np.ones(5)

        coeff = phys_ridge.fit(X, y)
        pred = X.dot(coeff)
        self.assertTrue(np.allclose(y, pred, atol=1e-3))

        A = np.array([[0.0, 1.0, 1.0, 0.0, 0.0]])
        c = np.zeros(1)
        phys_ridge.add_constraint(A, c)
        coeff = phys_ridge.fit(X, y)
        pred = X.dot(coeff)
        self.assertTrue(np.allclose(y, pred, atol=1e-3))

        self.assertAlmostEqual(coeff[1], -coeff[2])

    def test_non_constant_penalization(self):
        phys_ridge = PhysicalRidge(
            lamb_size=1.0, lamb_dia=1.0, size_decay="linear",
            dia_decay="linear", normalize=False)
        tests = [
            {
                'X': np.array([[1.0, 2.0], [-1.0, 3.0]]),
                'y': np.ones(2),
                'sizes': [2, 4],
                'diameters': [0.0, 0.0],
                'expect': np.array([5.0/67.0, 20.0/67.0])
            },
            {
                'X': np.array([[1.0, 2.0, -3.0], [-1.0, 3.0, 6.0]]),
                'y': np.ones(2),
                'expect': np.array([59.0/706.0, 209.0/706.0, 3.0/706.0]),
                'sizes': [2, 4, 2],
                'diameters': [0.0, 0.0, 0.0]
            },
             {
                'X': np.array([[1.0, 2.0], [-1.0, 3.0], [-5.0, 8.0]]),
                'y': np.ones(3),
                'expect': np.array([32.0/167.0, 43.0/167.0]),
                'sizes': [2, 4],
                'diameters': [0.0, 0.0]
            }
        ]

        for test in tests:
            phys_ridge.sizes = test['sizes']
            phys_ridge.diameters = test['diameters']
            coeff = phys_ridge.fit(test['X'], test['y'])
            self.assertTrue(np.allclose(coeff, test['expect']))


if __name__ == '__main__':
    unittest.main()
