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



if __name__ == '__main__':
    unittest.main()