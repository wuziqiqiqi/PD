import pytest
import numpy as np
from clease.regression import PhysicalRidge
from clease.regression.physical_ridge import random_cv_hyper_opt


def test_size_from_name():
    phys_ridge = PhysicalRidge()
    names = ['c0', 'c1_1', 'c2_d0000_0_00', 'c3_d1223_0_11']
    phys_ridge.sizes_from_names(names)
    expect = [0, 1, 2, 3]
    assert expect == phys_ridge.sizes


def test_dia_from_name():
    phys_ridge = PhysicalRidge()
    names = ['c0', 'c1_1', 'c2_d0000_0_00', 'c3_d1223_0_11']
    phys_ridge.diameters_from_names(names)
    expect = [0, 0, 0, 1223]
    assert phys_ridge.diameters == expect


def test_fit():
    phys_ridge = PhysicalRidge()
    names = ['c0', 'c1_1', 'c2_d0000_0_00', 'c3_d0002_0_11']

    X = np.random.rand(10, 4)
    X[:, 0] = 1.0
    y = np.random.rand(10)

    with pytest.raises(ValueError):
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


@pytest.mark.parametrize('normalize', [True, False])
def test_normalize(normalize):
    X = np.array([[1.0, 0.3, 0.5, 0.6], [1.0, -0.2, 0.7, 0.9], [1.0, -0.6, 0.3, 0.8],
                  [1.0, 0.2, 0.6, 1.2]])
    y = np.array([0.4, 0.2, -0.1, -0.6])

    phys_ridge = PhysicalRidge(normalize=normalize)
    phys_ridge.sizes = [1, 2, 3, 4]
    phys_ridge.diameters = [1, 2, 3, 4]
    coeff = phys_ridge.fit(X, y)
    pred = X.dot(coeff)

    # Allow some tolerence since we don't expect perfect match because
    # of the regularization
    assert np.allclose(y, pred, atol=1e-4), f"Normalize: {normalize}"


def test_constraints():
    phys_ridge = PhysicalRidge(lamb_dia=0.0, lamb_size=1e-4, normalize=False)
    X = np.zeros((3, 5))
    x = np.array([0.0, 2.0, 4.0])
    for i in range(5):
        X[:, i] = x**i
    y = 2.0 * x + x**2
    phys_ridge.diameters = np.zeros(5)
    phys_ridge.sizes = 2 * np.ones(5)

    coeff = phys_ridge.fit(X, y)
    pred = X.dot(coeff)
    assert np.allclose(y, pred, atol=1e-3)

    A = np.array([[0.0, 1.0, 1.0, 0.0, 0.0]])
    c = np.zeros(1)
    phys_ridge.add_constraint(A, c)
    coeff = phys_ridge.fit(X, y)
    pred = X.dot(coeff)
    assert np.allclose(y, pred, atol=1e-3)

    assert coeff[1] == pytest.approx(-coeff[2])


@pytest.mark.parametrize('test', [
    {
        'X': np.array([[1.0, 2.0], [-1.0, 3.0]]),
        'y': np.ones(2),
        'sizes': [2, 4],
        'diameters': [0.0, 0.0],
        'expect': np.array([5.0 / 67.0, 20.0 / 67.0])
    },
    {
        'X': np.array([[1.0, 2.0, -3.0], [-1.0, 3.0, 6.0]]),
        'y': np.ones(2),
        'expect': np.array([59.0 / 706.0, 209.0 / 706.0, 3.0 / 706.0]),
        'sizes': [2, 4, 2],
        'diameters': [0.0, 0.0, 0.0]
    },
    {
        'X': np.array([[1.0, 2.0], [-1.0, 3.0], [-5.0, 8.0]]),
        'y': np.ones(3),
        'expect': np.array([32.0 / 167.0, 43.0 / 167.0]),
        'sizes': [2, 4],
        'diameters': [0.0, 0.0]
    },
])
def test_non_constant_penalization(test):
    phys_ridge = PhysicalRidge(lamb_size=1.0,
                               lamb_dia=1.0,
                               size_decay="linear",
                               dia_decay="linear",
                               normalize=False)
    phys_ridge.sizes = test['sizes']
    phys_ridge.diameters = test['diameters']
    coeff = phys_ridge.fit(test['X'], test['y'])
    assert np.allclose(coeff, test['expect'])
