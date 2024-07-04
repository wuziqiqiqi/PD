import pytest
import numpy as np
from clease.data_normalizer import DataNormalizer


def test_normalizer():
    X = np.array([[1.0, -4.0], [-1.0, 2.0], [1.0, 2.0]])

    coeff = [1.0, -2.0]
    y = X.dot(coeff)
    meanY = np.mean(y)
    meanX = [1.0 / 3, 0.0]
    stdX = [2.0 / np.sqrt(3.0), np.sqrt(12.0)]

    normalizer = DataNormalizer()
    X_norm, y_norm = normalizer.normalize(X, y)

    assert np.allclose(normalizer.meanX, meanX)
    assert np.allclose(normalizer.stdX, stdX)
    assert normalizer.meanY == pytest.approx(meanY)

    coeff_orig, _, _, _ = np.linalg.lstsq(X, y, rcond=-1)
    coeff_norm, _, _, _ = np.linalg.lstsq(X_norm, y_norm, rcond=-1)
    converted = normalizer.convert(coeff_norm)
    assert np.allclose(coeff_orig, converted, rtol=0.2)
    assert normalizer.bias(coeff_norm) == pytest.approx(0.0)
