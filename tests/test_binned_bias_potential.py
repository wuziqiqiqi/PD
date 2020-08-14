import pytest
from unittest.mock import MagicMock
import numpy as np
from clease.montecarlo import BinnedBiasPotential


def bias_12_11bins(getter=None):
    return BinnedBiasPotential(xmin=1.0, xmax=2.0, nbins=10, getter=getter)


def test_get_index():
    bias = bias_12_11bins()
    i = bias.get_index(1.5)
    assert i == 5


def test_get_x():
    bias = bias_12_11bins()
    x = bias.get_x(5)
    assert x == pytest.approx(1.55)


def test_evaluate():
    bias = BinnedBiasPotential(xmin=0.0, xmax=2.0, nbins=10000)
    x = np.array([bias.get_x(i) for i in range(bias.nbins)])
    bias.values = x**2

    for i, x in enumerate([bias.dx / 2, 1.3, 2.0 - bias.dx / 2]):
        y = bias.evaluate(x)
        assert y == pytest.approx(x**2), 'Failed test #{}: Expected: {}, got: {}'.format(i, x**2, y)


def test_call():

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
        assert y == pytest.approx(test[1]), 'Failed for test #{}: Expected: {}, got: {}'.format(
            i, test[1], y)


def test_to_from_dict():
    bias = BinnedBiasPotential(xmin=0.0, xmax=1.0, nbins=10)
    data = bias.todict()
    bias2 = BinnedBiasPotential(xmin=1.0, xmax=2.0, nbins=15)
    bias2.from_dict(data)
    assert bias2.nbins == bias.nbins
    assert bias2.dx == pytest.approx(bias.dx)
    assert bias2.xmin == pytest.approx(bias.xmin)
    assert bias2.xmax == pytest.approx(bias.xmax)
    assert np.allclose(bias2.values, bias.values)


def test_calc_from_scratch():
    bias = bias_12_11bins(getter=MagicMock())
    bias.calculate_from_scratch(None)
    bias.getter.calculate_from_scratch.assert_called_with(None)


def test_local_update():
    bias = bias_12_11bins()
    bias.local_update(1.55, 0.6)
    y = bias.evaluate(1.55)
    assert y == pytest.approx(0.6)
