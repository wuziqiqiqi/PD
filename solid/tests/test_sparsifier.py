import numpy as np
import matplotlib
from clease.sparsifier import Sparsifier
from clease.regression import LinearRegression


def test_sparsifier():
    matplotlib.use("Agg")
    num_feat = 12
    num_data = 100
    X = np.zeros((num_data, num_feat))
    x = np.linspace(0.0, 10.0, num_data)
    for i in range(num_feat):
        X[:, i] = x**i

    y = 2.0 * X[:, 5] - 7.0 * X[:, 9]

    sparsifier = Sparsifier()
    fitter = LinearRegression()
    selection, coeff = sparsifier.sparsify(fitter, X, y)
    sparsifier.plot()

    assert selection == [5, 9]
    assert np.allclose(coeff, [2.0, -7.0])
