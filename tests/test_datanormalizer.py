import unittest
from clease.data_normalizer import DataNormalizer
import numpy as np


class TestDataNormalizer(unittest.TestCase):
    def test_normalizer(self):
        X = np.array([[1.0, -4.0],
                      [-1.0, 2.0],
                      [1.0, 2.0]])

        coeff = [1.0, -2.0]
        y = X.dot(coeff)
        meanY = np.mean(y)
        meanX = [1./3, 0.0]
        stdX = [2.0/np.sqrt(3.0), np.sqrt(12.0)]

        normalizer = DataNormalizer()
        X_norm, y_norm = normalizer.normalize(X, y)

        self.assertTrue(np.allclose(normalizer.meanX, meanX))
        self.assertTrue(np.allclose(normalizer.stdX, stdX))
        self.assertAlmostEqual(normalizer.meanY, meanY)

        coeff_orig, _, _, _ = np.linalg.lstsq(X, y)
        coeff_norm, _, _, _ = np.linalg.lstsq(X_norm, y_norm)
        converted = normalizer.convert(coeff_norm)
        self.assertTrue(np.allclose(coeff_orig, converted, rtol=0.2))
        self.assertAlmostEqual(normalizer.bias(coeff_norm), 0.0)


if __name__ == '__main__':
    unittest.main()
