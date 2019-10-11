import unittest
from unittest.mock import MagicMock
from clease.montecarlo import BinnedBiasPotential
import numpy as np


class TestBinnedBiasPotential(unittest.TestCase):
    def bias_12_11bins(self, getter=None):
        return BinnedBiasPotential(xmin=1.0, xmax=2.0, nbins=10, getter=getter)

    def test_get_index(self):
        bias = self.bias_12_11bins()
        i = bias.get_index(1.5)
        self.assertEqual(i, 5)

    def test_get_x(self):
        bias = self.bias_12_11bins()
        x = bias.get_x(5)
        self.assertAlmostEqual(1.5, x)

    def test_evaluate(self):
        bias = BinnedBiasPotential(xmin=0.0, xmax=2.0, nbins=10000)
        x = np.array([bias.get_x(i) for i in range(bias.nbins)])
        bias.values = x**2

        for i, x in enumerate([bias.dx/2, 1.3, 2.0-bias.dx/2]):
            y = bias.evaluate(x)
            self.assertAlmostEqual(
                y, x**2, msg='Failed test #{}: Expected: {}, got: {}'
                             ''.format(i, x**2, y))

    def test_call(self):
        def getter(syst_change, peak=False):
            if syst_change[0][1] == 'Al':
                return 0.5
            return 0.4

        bias = BinnedBiasPotential(xmin=0.0, xmax=1.0, nbins=11, getter=getter)
        x = np.array([bias.get_x(i) for i in range(bias.nbins)])
        bias.values = x

        changes = [[(1, 'Al', 'Mg')], [(2, 'Mg', 'Al')]]
        expect = [0.5, 0.4]
        for i, test in enumerate(zip(changes, expect)):
            y = bias(test[0])
            self.assertAlmostEqual(
                y, test[1], msg='Failed for test #{}: Expected: {}, got: {}'
                                ''.format(i, test[1], y))

    def test_to_from_dict(self):
        bias = BinnedBiasPotential(xmin=0.0, xmax=1.0, nbins=10)
        data = bias.todict()
        bias2 = BinnedBiasPotential(xmin=1.0, xmax=2.0, nbins=15)
        bias2.from_dict(data)
        self.assertEqual(bias2.nbins, bias.nbins)
        self.assertAlmostEqual(bias2.dx, bias.dx)
        self.assertAlmostEqual(bias2.xmin, bias.xmin)
        self.assertAlmostEqual(bias2.xmax, bias.xmax)
        self.assertTrue(np.allclose(bias2.values, bias.values))

    def test_calc_from_scratch(self):
        bias = self.bias_12_11bins(getter=MagicMock())
        bias.calculate_from_scratch(None)
        bias.getter.calculate_from_scratch.assert_called_with(None)

    def test_local_update(self):
        bias = self.bias_12_11bins()
        bias.local_update(1.5, 0.6)
        y = bias.evaluate(1.5)
        self.assertAlmostEqual(y, 0.6)


if __name__ == '__main__':
    unittest.main()
