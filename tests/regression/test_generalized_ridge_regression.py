import pytest
from cleases.regression import GeneralizedRidgeRegression, EigenDecomposition
import numpy as np


def test_consistent_gradients():
    X = np.zeros((4, 5))
    y = np.zeros(4)
    x = np.linspace(0.0, 2.0, 4)
    for i in range(X.shape[1]):
        X[:, i] = x**i
    y = 2.0 * x + 6.0 * x**3

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
        fd_grad = (new_gcv - gcv_sq) / delta
        assert fd_grad == pytest.approx(grad[i], abs=1e-4), f"Component #{i}"


def test_fit():
    X = np.zeros((4, 5))
    y = np.zeros(4)
    x = np.linspace(0.5, 2.0, 4)
    for i in range(X.shape[1]):
        X[:, i] = x**i
    y = 2.0 * x + 6.0 * x**3

    alpha = np.ones(X.shape[1]) * 1e-8
    ridge = GeneralizedRidgeRegression(alpha=alpha)
    coeff = ridge.fit(X, y)
    pred = X.dot(coeff)
    assert np.allclose(y, pred, atol=1e-5)
