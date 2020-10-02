import pytest

from clease.regression import Tikhonov


@pytest.mark.parametrize('test', [
    {
        'alpha': 10.0,
        'expect': True
    },
    {
        'alpha': 5.0,
        'expect': True
    },
    {
        'alpha': [20, 20, 10],
        'expect': False
    },
    {
        'alpha': 10,
        'expect': False
    },
])
def test_isscalar(test):
    tik = Tikhonov()

    tik.alpha = test['alpha']
    assert tik.alpha == test['alpha']
    assert test['expect'] == tik.is_scalar()
