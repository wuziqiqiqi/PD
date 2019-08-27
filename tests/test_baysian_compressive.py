import os
import unittest
from clease import BayesianCompressiveSensing
import numpy as np
from scipy.special import polygamma

# Fix the seed to ensure consistent tests
np.random.seed(0)
fname = "test_bayes_compr_sens.json"


class TestBayesianCompressiveSensing(unittest.TestCase):
    bayes = BayesianCompressiveSensing(fname=fname, output_rate_sec=2,
                                       maxiter=100)

    def test_optimize_shape_parameter(self):
        self.bayes.lamb = 1.0
        opt = self.bayes.optimal_shape_lamb()
        self.assertAlmostEqual(np.log(opt/2.0), polygamma(0, opt/2.0))

    def test_fit(self):
        X = np.random.rand(30, 400)
        y = 60.0*X[:, 20] - 80.0*X[:, 2]
        eci = self.bayes.fit(X, y)

        expected_eci = np.zeros(X.shape[1])
        expected_eci[20] = 60.0
        expected_eci[2] = -80.0
        self.assertTrue(np.allclose(eci, expected_eci, rtol=1E-4))

    def test_fit_more_coeff(self):
        self.bayes = BayesianCompressiveSensing(fname=fname, noise=0.1,
                                                maxiter=1000)
        X = np.random.rand(30, 400)
        coeff = [6.0, -2.0, 5.0, 50.0, -30.0]
        indx = [0, 23, 19, 18, 11]
        y = 0.0
        expected_eci = np.zeros(X.shape[1])
        for c, i in zip(coeff, indx):
            y += X[:, i]*c
            expected_eci[i] = c
        eci = self.bayes.fit(X, y)
        self.assertTrue(np.allclose(eci, expected_eci, atol=1E-2))

    def test_save_load(self):
        self.bayes.save()
        bayes2 = BayesianCompressiveSensing.load(fname)
        self.assertTrue(self.bayes == bayes2)

    def test_fit_linear_dep_col(self):
        bayes = BayesianCompressiveSensing(fname=fname, noise=0.2,
                                           penalty=1E-2, maxiter=1000)
        X = np.random.rand(30, 400)
        X[:, 2] = X[:, 0]
        X[:, 8] = X[:, 20]
        X[:, 23] = X[:, 50]
        y = 20*X[:, 0] - 3*X[:, 23]
        eci = bayes.fit(X, y)
        self.assertEqual(len(eci), 400)

        expected = np.zeros(400)
        expected[0] = 20
        expected[23] = -3
        print(np.argwhere(np.abs(eci) > 1E-8))
        self.assertTrue(np.allclose(eci, expected, atol=1E-2))

        prec = bayes.precision_matrix(X)
        self.assertTrue(prec.shape == (400, 400))

    def tearDown(self):
        try:
            os.remove(fname)
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
