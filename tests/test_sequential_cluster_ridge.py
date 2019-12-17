import unittest
import numpy as np
from clease import SequentialClusterRidge


class TestSequentialClusterRidge(unittest.TestCase):
    def test_support_expanding(self):
        X = np.zeros((200, 10))
        x = np.linspace(0.0, 2.0, 200)
        for i in range(10):
            X[:, i] = x**i

        y = 5.0*x**2 - 2.0*x**6

        fit_scheme = SequentialClusterRidge(
            min_alpha=1e-16, max_alpha=1e-4, num_alpha=20)

        coeff = fit_scheme.fit(X, y)
        expected = np.zeros(10)
        expected[2] = 5.0
        expected[6] = -2.0
        self.assertGreater(len(coeff), 6)
        self.assertTrue(np.allclose(expected[:len(coeff)], coeff, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
