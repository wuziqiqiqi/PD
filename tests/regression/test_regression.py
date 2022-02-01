import pytest
import numpy as np
from clease.regression import LinearRegression, Lasso, Tikhonov


@pytest.fixture
def x():
    return np.linspace(0.0, 1.0, 20)


@pytest.fixture
def y(x):
    return 1.0 + 2.0 * x - 4.0 * x**2


def test_non_singular(x, y):
    X = np.zeros((len(x), 3))
    X[:, 0] = 1.0
    X[:, 1] = x
    X[:, 2] = x**2

    linreg = LinearRegression()
    coeff = linreg.fit(X, y)

    # Test that fit works
    assert np.allclose(coeff, [1.0, 2.0, -4.0])

    # Test that precision matrix gives correct result
    # in the case where it is not singular
    prec = np.linalg.inv(X.T.dot(X))
    prec_regr = linreg.precision_matrix(X)
    assert np.allclose(prec, prec_regr)


def test_trivial_singular(x, y):
    X = np.zeros((len(x), 4))
    X[:, 0] = 1.0
    X[:, 1] = x
    X[:, 2] = x**2
    X[:, 3] = x**2

    linreg = LinearRegression()
    coeff = linreg.fit(X, y)

    assert np.allclose(X.dot(coeff), y)

    # Check that it is actually singular
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(X.T.dot(X))

    # Check we can evaluate
    linreg.precision_matrix(X)


def test_complicated_singular(x, y):
    X = np.zeros((len(x), 5))
    X[:, 0] = 1.0
    X[:, 1] = x
    X[:, 2] = x**2
    X[:, 3] = 0.1 - 0.2 * x + 0.8 * x**2
    X[:, 4] = -0.2 + 0.8 * x

    linreg = LinearRegression()
    coeff = linreg.fit(X, y)

    assert np.allclose(X.dot(coeff), y)
    linreg.precision_matrix(X)


@pytest.mark.parametrize("reg_func", (Lasso, Tikhonov))
@pytest.mark.parametrize(
    "test",
    [
        {"alpha_min": 0.2, "alpha_max": 0.5, "num_alpha": 5, "scale": "log"},
        {"alpha_min": 1, "alpha_max": 5, "num_alpha": 10, "scale": "log"},
        {"alpha_min": 2, "alpha_max": 4, "num_alpha": 4, "scale": "etc"},
    ],
)
def test_get_instance_array(reg_func, test):
    reg = reg_func()
    if test["scale"] == "log":
        true_alpha = np.logspace(
            np.log10(test["alpha_min"]),
            np.log10(test["alpha_max"]),
            test["num_alpha"],
            endpoint=True,
        )
    else:
        true_alpha = np.linspace(
            test["alpha_min"], test["alpha_max"], int(test["num_alpha"]), endpoint=True
        )

    instance_array = reg.get_instance_array(
        test["alpha_min"], test["alpha_max"], test["num_alpha"], test["scale"]
    )
    predict_alpha = np.array([i.alpha for i in instance_array])

    assert predict_alpha.shape == true_alpha.shape
    assert np.allclose(true_alpha, predict_alpha)
