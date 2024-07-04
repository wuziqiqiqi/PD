import pytest
import numpy as np
from cleases.regression import ConstrainedRidge


@pytest.fixture
def system():
    """Simple test system for fitting"""
    X = np.zeros((3, 5))
    x = np.linspace(0.0, 1.0, 3)
    for i in range(X.shape[1]):
        X[:, i] = x**i

    y = 2.0 * x - 3.0 * x**3
    return X, y


@pytest.fixture
def regressor():
    alpha = 1e-5
    alpha_vec = np.zeros(5) + alpha
    reg = ConstrainedRidge(alpha_vec)
    return reg


def test_constrained_ridge(regressor, system):
    X, y = system

    coeff = regressor.fit(X, y)
    pred = X.dot(coeff)
    assert np.allclose(y, pred, atol=1e-3)


def test_constrained_constrained_ridge(regressor, system):
    X, y = system
    # Apply a constraint the first coefficient is
    # two times the second
    A = np.array([[0.0, -1.0, 2.0, 0.0, 0.0]])
    c = np.zeros(1)

    # We know have five unknowns, three data points and one constraint
    # Thus, there is still one additional degree of freedom. Thys,
    # all data points should still be fitted accurately
    regressor.add_constraint(A, c)
    coeff = regressor.fit(X, y)
    pred = X.dot(coeff)
    assert np.allclose(y, pred, atol=1e-3)

    # Make sure the constrain it satisfied
    assert coeff[1] == pytest.approx(2 * coeff[2])
