from clease import LinearRegression
import numpy as np
import unittest


class TestRegression(unittest.TestCase):
    def test_non_singular(self):
        x = np.linspace(0.0, 1.0, 20)

        y = 1.0 + 2.0*x - 4.0*x**2

        X = np.zeros((len(x), 3))
        X[:, 0] = 1.0
        X[:, 1] = x
        X[:, 2] = x**2

        linreg = LinearRegression()
        coeff = linreg.fit(X, y)

        # Test that fit works
        self.assertTrue(np.allclose(coeff, [1.0, 2.0, -4.0]))

        # Test that precision matrix gives correct result
        # in the case where it is not singular
        prec = np.linalg.inv(X.T.dot(X))
        prec_regr = linreg.precision_matrix(X)
        self.assertTrue(np.allclose(prec, prec_regr))

    def test_trivial_singular(self):
        x = np.linspace(0.0, 1.0, 20)

        y = 1.0 + 2.0*x - 4.0*x**2

        X = np.zeros((len(x), 4))
        X[:, 0] = 1.0
        X[:, 1] = x
        X[:, 2] = x**2
        X[:, 3] = x**2

        linreg = LinearRegression()
        coeff = linreg.fit(X, y)

        self.assertTrue(np.allclose(X.dot(coeff), y))
        linreg.precision_matrix(X)

    def test_complicated_singular(self):
        x = np.linspace(0.0, 1.0, 20)

        y = 1.0 + 2.0*x - 4.0*x**2

        X = np.zeros((len(x), 5))
        X[:, 0] = 1.0
        X[:, 1] = x
        X[:, 2] = x**2
        X[:, 3] = 0.1 - 0.2*x + 0.8*x**2
        X[:, 4] = -0.2 + 0.8*x

        linreg = LinearRegression()
        coeff = linreg.fit(X, y)

        self.assertTrue(np.allclose(X.dot(coeff), y))
        linreg.precision_matrix(X)


if __name__ == '__main__':
    unittest.main()
