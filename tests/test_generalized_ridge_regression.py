import unittest
from clease import GeneralizedRidgeRegression, EigenDecomposition
import numpy as np


class TestGeneralizedRidgeRegression(unittest.TestCase):
    def test_consistent_gradients(self):
        X = np.zeros((4, 5))
        y = np.zeros(4)
        x = np.linspace(0.0, 2.0, 4)
        for i in range(X.shape[1]):
            X[:, i] = x**i
        y = 2.0*x + 6.0*x**3

        alpha = np.array([1.4, 2.0, 3.0, 4.0, 5.0])
        ridge = GeneralizedRidgeRegression(alpha)
        eigen = EigenDecomposition()
        eigen.decompose(X.T.dot(X))

        grad = ridge._grad_gcv_squared(alpha, X, y, eigen)

        # Calculate derivatives with finite differences
        gcv_sq = ridge._gcv_squared(alpha, X, y, eigen)
        for i in range(len(alpha)):
            delta = 0.0001
            alpha_tmp = alpha.copy()
            alpha_tmp[i] += delta
            new_gcv = ridge._gcv_squared(alpha_tmp, X, y, eigen)
            fd_grad = (new_gcv - gcv_sq)/delta
            self.assertAlmostEqual(fd_grad, grad[i], places=4,
                                   msg=f"Component #{i}")

    def test_fit(self):
        X = np.zeros((4, 5))
        y = np.zeros(4)
        x = np.linspace(0.0, 2.0, 4)
        for i in range(X.shape[1]):
            X[:, i] = x**i
        y = 2.0*x + 6.0*x**3
        ridge = GeneralizedRidgeRegression()
        coeff = ridge.fit(X, y)
        pred = X.dot(coeff)
        self.assertTrue(np.allclose(y, pred))


if __name__ == '__main__':
    unittest.main()            
