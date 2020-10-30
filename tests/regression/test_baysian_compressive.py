from copy import deepcopy
import pytest
import numpy as np
from scipy.special import polygamma
from clease.regression import BayesianCompressiveSensing

# Fix the seed to ensure consistent tests
np.random.seed(0)


@pytest.fixture
def make_bayes(make_tempfile):
    """Factory fixture for making Bayes class"""
    filename = "test_bayes_compr_sens.json"
    fname = make_tempfile(filename)

    def _make_bayes(**kwargs):
        default = dict(fname=fname, output_rate_sec=2, maxiter=100)
        default.update(kwargs)
        return BayesianCompressiveSensing(**default)

    return _make_bayes


@pytest.fixture
def bayes(make_bayes):
    """Default (uninitialized) bayes instance"""
    res = make_bayes()
    # We should be uninitialized
    assert res.gammas is None
    return res


@pytest.fixture
def bayes_initialized(bayes):
    """Bayes initialized on random data"""
    bayes = deepcopy(bayes)  # Ensure we aren't overriding the other fixture
    assert bayes.gammas is None
    X = np.random.rand(30, 400)
    y = np.random.rand(30)
    bayes.fit(X, y)
    assert isinstance(bayes.gammas, np.ndarray)
    return bayes


def test_optimize_shape_parameter(bayes_initialized):
    bayes = bayes_initialized  # Rename
    bayes.lamb = 1.0
    opt = bayes.optimal_shape_lamb()
    assert np.log(opt / 2.0) == pytest.approx(polygamma(0, opt / 2.0))


def test_fit(bayes):
    X = np.random.rand(30, 400)
    y = 60.0 * X[:, 20] - 80.0 * X[:, 2]
    eci = bayes.fit(X, y)

    expected_eci = np.zeros(X.shape[1])
    expected_eci[20] = 60.0
    expected_eci[2] = -80.0
    assert np.allclose(eci, expected_eci, rtol=1E-4)


def test_fit_more_coeff(make_bayes):
    # Adjust some parameters
    bayes = make_bayes(noise=0.1, maxiter=1000)

    np.random.seed(42)
    X = np.random.rand(30, 400)
    coeff = [6.0, -2.0, 5.0, 50.0, -30.0]
    indx = [0, 23, 19, 18, 11]
    y = 0.0
    expected_eci = np.zeros(X.shape[1])
    for c, i in zip(coeff, indx):
        y += X[:, i] * c
        expected_eci[i] = c
    eci = bayes.fit(X, y)
    assert np.allclose(eci, expected_eci, atol=1E-2)


def test_to_dict(bayes, bayes_initialized):
    # Variables we expect to find in the dictionaries
    vars_to_save = ("inv_variance", 'gammas', 'shape_var', 'rate_var', 'shape_lamb', 'lamb',
                    'maxiter', 'output_rate_sec', 'select_strategy', 'noise', 'lamb_opt_start')

    dct1 = bayes.to_dict()
    dct2 = bayes_initialized.to_dict()
    for var in vars_to_save:
        assert var in dct1
        assert var in dct2


def test_save_load(bayes, bayes_initialized):

    def save_load(bayes1):
        fname = bayes1.fname
        bayes1.save()
        bayes2 = BayesianCompressiveSensing.load(fname)
        assert bayes1 == bayes2

    # Save an uninitialized instance
    save_load(bayes)

    # Now we do a fit to initialize constants
    save_load(bayes_initialized)


def test_fit_linear_dep_col(make_bayes):
    bayes = make_bayes(noise=0.2, penalty=1E-2, maxiter=1000)
    X = np.random.rand(30, 400)
    X[:, 2] = X[:, 0]
    X[:, 8] = X[:, 20]
    X[:, 23] = X[:, 50]
    y = 20 * X[:, 0] - 3 * X[:, 23]
    eci = bayes.fit(X, y)
    assert len(eci) == 400

    pred = X.dot(eci)
    assert np.allclose(pred, y, atol=1e-2)

    prec = bayes.precision_matrix(X)
    assert prec.shape == (400, 400)
