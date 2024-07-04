import pytest
import random
import numpy as np
from clease.datastructures import TransMatrix


@pytest.fixture
def make_random_trans_matrix():
    def _maker(n_sites=10, n_indices=8, background_indices=None):
        keys = list(range(n_indices))
        values = list(keys)
        tm = []
        for i in range(n_sites):
            if i > 0:
                random.shuffle(values)
            dct = dict(zip(keys, values))
            tm.append(dct)
        if background_indices:
            for idx in background_indices:
                tm[idx] = {}
        return TransMatrix(tm)

    return _maker


def test_tm(make_random_trans_matrix):
    for sites in range(15):
        for indices in range(6):
            tm = make_random_trans_matrix(sites, indices)
            assert isinstance(tm, TransMatrix)
            assert isinstance(tm.trans_matrix, list)
            assert all(isinstance(dct, dict) for dct in tm)
            assert len(tm) == sites
            assert len(tm) == len(tm.trans_matrix)


def test_array_roundstrip(make_random_trans_matrix):
    for sites in range(15):
        for indices in range(6):
            tm = make_random_trans_matrix(sites, indices)
            arr = tm.as_array()
            assert len(arr) == len(tm)
            tm2 = TransMatrix.from_array(arr)
            assert tm == tm2
            keys = tm.key_array()
            values = tm.value_array()

            assert np.array_equal(arr[:, :, 0], keys)
            assert np.array_equal(arr[:, :, 1], values)


def test_array_background(make_random_trans_matrix):
    for sites in range(4, 15):
        for indices in range(2, 8):
            tm = make_random_trans_matrix(sites, indices, background_indices=[2, 3])
            arr = tm.as_array()
            assert len(arr) == len(tm)
            # Verify the background sites
            assert (arr[2, :, :] == -1).all()
            assert (arr[3, :, :] == -1).all()
            assert tm[2] == {}
            assert tm[3] == {}
            tm2 = TransMatrix.from_array(arr)
            assert tm == tm2
            keys = tm.key_array()
            values = tm.value_array()

            assert np.array_equal(arr[:, :, 0], keys)
            assert np.array_equal(arr[:, :, 1], values)
