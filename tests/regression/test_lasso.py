import pytest
import sklearn.linear_model
import numpy as np
from clease.regression import Lasso


@pytest.mark.parametrize('data', [{
    'X': [[1, 2], [3, 4], [5, 3]],
    'Y': [4, 5, 2]
}, {
    'X': [[3, 1, 5], [2, 4, 4], [6, 5, 3]],
    'Y': [4, 5, 2]
}, {
    'X': [[1, 2, 6, 1], [3, 4, 3, 8], [5, 3, 4, 6]],
    'Y': [9, 1, 2]
}, {
    'X': [[1, 2, 12, 43, 2], [3, 4, 3, 12, 21], [5, 3, 21, 17, 20]],
    'Y': [4, 5, 2]
}, {
    'X': [[1, 2, 6, 1, 0, 12], [3, 4, 6, 7, 8, 12], [5, 3, 3, 4, 5, 6]],
    'Y': [10, 31, 12]
}])
def test_fit(data):
    sk_lasso = sklearn.linear_model.Lasso(alpha=1, fit_intercept=False, copy_X=True, max_iter=1e6)
    cl_lasso = Lasso(alpha=1)

    sk_lasso.fit(data['X'], data['Y'])
    expect = sk_lasso.coef_
    predict = cl_lasso.fit(data['X'], data['Y'])

    assert np.allclose(expect, predict)


@pytest.mark.parametrize('test', [{
    'alpha': 10.0,
    'bool': True
}, {
    'alpha': 5.0,
    'bool': True
}, {
    'alpha': [20, 20, 10],
    'bool': False
}, {
    'alpha': 10,
    'bool': False
}])
def test_get_scalar_parameter(test):
    las = Lasso()
    assert las.is_scalar()
    las.alpha = test['alpha']
    assert las.get_scalar_parameter() == test['alpha']
