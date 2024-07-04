import numpy as np
from clease.svd import SVD


def test_svd():
    X = [[2, 2, 2], [1, -1, -1]]

    svd = SVD()
    assert not svd.has_matrices()
    svd.calculate(X)
    assert svd.has_matrices()

    U = [[-0.978216, 0.207591], [0.207591, 0.978216]]

    V = [[-0.496149, 0.868238], [-0.613937, -0.35083], [-0.613937, -0.35083]]
    Vh = np.array(V).T
    S = [3.52483, 1.60486]

    assert np.allclose(svd.U, U)
    assert np.allclose(svd.Vh, Vh)
    assert np.allclose(svd.S, S)
