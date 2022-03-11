import pytest
from unittest.mock import MagicMock
import numpy as np
from clease.datastructures import SystemChange
from clease.montecarlo.gaussian_kernel_bias_potential import GaussianKernelBiasPotential


def assertAlmostEqual(a, b):
    """Helper function for porting unittest to pytest"""
    assert a == pytest.approx(b)


def test_in_range():
    pot = GaussianKernelBiasPotential(xmin=0.0, xmax=1.0, num_kernels=100, width=0.05)

    x = 0.64
    lower, upper = pot.inside_range(x)
    kernels = pot._gaussian(x, pot.centers)
    mask = np.ones(len(kernels), dtype=np.uint8)
    mask[mask >= lower] = 0
    mask[mask <= upper] = 0
    assert np.all(kernels[mask] < 0.01)


def test_evaluate():
    def get_func(syst_change, peak=False):
        if syst_change[0].new_symb == "Al":
            return 1.03
        return 0.0

    pot = GaussianKernelBiasPotential(
        xmin=0.0, xmax=1.0, num_kernels=10, width=0.01, getter=get_func
    )
    pot.coeff[:] = np.linspace(0.0, 10.0, len(pot.coeff))
    expect = 10.0
    got = pot([SystemChange(0, "Mg", "Al")])
    assert got == pytest.approx(expect)

    expect = 0.0
    got = pot([SystemChange(0, "Al", "Mg")])
    assert got == pytest.approx(expect)


def test_get_index():
    pot = GaussianKernelBiasPotential(xmin=0.0, xmax=1.0, num_kernels=11, width=0.05)
    index = pot.get_index(0.5)
    assert index == 5 + pot.pad


def test_get_x():
    pot = GaussianKernelBiasPotential(xmin=0.0, xmax=1.0, num_kernels=11, width=0.05)
    x = pot.get_x(5 + pot.pad)
    assert x == pytest.approx(0.5)


def test_local_update():
    pot = GaussianKernelBiasPotential(xmin=0.0, xmax=1.0, num_kernels=100, width=0.1)

    dE = 4.0
    x = 0.64
    pot.local_update(x, dE)
    got = pot.evaluate(x)
    assert dE == pytest.approx(got)


def test_to_from_dict():
    xmin = -5.0
    xmax = 2.0
    num_kernels = 10
    width = 0.4
    pot = GaussianKernelBiasPotential(xmin=xmin, xmax=xmax, num_kernels=num_kernels, width=width)
    xmax_corr = pot.xmax_corrected
    xmin_corr = pot.xmin_corrected
    num_kernels = pot.num_kernels  # There are some padding

    pot.coeff = np.linspace(0.0, 10.0, len(pot.coeff))
    centers = pot.centers.copy()
    coeff = pot.coeff.copy()

    data = pot.todict()

    # Initialise another object
    pot = GaussianKernelBiasPotential()
    pot.from_dict(data)

    # Confirm that all variables matches
    assertAlmostEqual(xmin, pot.xmin)
    assertAlmostEqual(xmax, pot.xmax)
    assertAlmostEqual(xmin_corr, pot.xmin_corrected)
    assertAlmostEqual(xmax_corr, pot.xmax_corrected)
    assertAlmostEqual(width, pot.width)
    assert num_kernels == pot.num_kernels
    assert np.allclose(pot.centers, centers)
    assert np.allclose(pot.coeff, coeff)


def test_calc_from_scratch():
    getter = MagicMock()
    getter.calculate_from_scratch.return_value = 0.5
    obs = GaussianKernelBiasPotential(getter=getter)
    obs.local_update(0.5, 2)
    value = obs.calculate_from_scratch(None)
    assertAlmostEqual(value, 2.0)


def test_ensure_zero_slope():
    pot = GaussianKernelBiasPotential()
    pot.local_update(0.01, 2)
    assert pot.slope(0.0) > 0.0
    pot.ensure_zero_slope(0.0)
    assertAlmostEqual(pot.slope(0.0), 0.0)
