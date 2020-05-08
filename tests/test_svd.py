import unittest
from clease.svd import SVD
import numpy as np


class TestSVD(unittest.TestCase):
    def test_svd(self):
        X = [[2, 2, 2],
             [1, -1, -1]]

        svd = SVD()
        self.assertFalse(svd.has_matrices())
        svd.calculate(X)
        self.assertTrue(svd.has_matrices())

        U = [[-0.978216, 0.207591], [0.207591, 0.978216]]

        V = [[-0.496149, 0.868238],
             [-0.613937, -0.35083],
             [-0.613937, -0.35083]]
        Vh = np.array(V).T
        S = [3.52483, 1.60486]

        self.assertTrue(np.allclose(svd.U, U))
        self.assertTrue(np.allclose(svd.Vh, Vh))
        self.assertTrue(np.allclose(svd.S, S))


if __name__ == '__main__':
    unittest.main()