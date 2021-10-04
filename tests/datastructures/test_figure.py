import pytest
import numpy as np
from clease.datastructures import FourVector, Figure


@pytest.fixture
def figure():
    fv1 = FourVector(0, 0, 0, 0)
    fv2 = FourVector(0, 0, 0, 1)
    return Figure({fv1, fv2})


def test_initialization():
    fv1 = FourVector(0, 0, 0, 0)
    fv2 = FourVector(0, 0, 0, 1)

    # Test we can pass in default iterable types such as Set, Tuple, List
    # Avoid testing equiality with set, since the order is non-deterministic
    fig_set = Figure({fv1, fv2})
    fig_tup = Figure((fv1, fv2))
    fig_lst = Figure([fv1, fv2])
    fig_arr = Figure(np.array([fv1, fv2]))  # Initialization from NumPy array

    assert fig_lst == fig_tup
    assert fig_arr == fig_lst


def test_equality():
    fv1 = FourVector(0, 0, 0, 0)
    fv2 = FourVector(0, 0, 0, 1)
    fv3 = FourVector(1, 0, 0, 0)

    # Different FourVectors
    assert Figure([fv1, fv2]) != Figure([fv2, fv3])
    # Different number of componenets
    assert Figure([fv1, fv2, fv3]) != Figure([fv1])
    # Order matters
    assert Figure([fv3, fv2, fv1]) != Figure([fv1, fv2, fv3])
    assert Figure([fv1, fv2, fv3]) == Figure([fv1, fv2, fv3])


def test_hashable(figure):
    hash(figure)
    s = set()
    s.add(figure)
    s.add(figure)
    assert len(s) == 1


def test_wrong_type():
    with pytest.raises(TypeError):
        # FourVector defined in its tuple form
        Figure([(0, 0, 0, 0)])

    with pytest.raises(TypeError):
        # Mixed FourVector and tuple form
        Figure([FourVector(0, 0, 0, 0), (0, 0, 0, 1)])

    with pytest.raises(TypeError):
        # Just a single FourVector not inside a list or anything
        Figure(FourVector(0, 0, 0, 0))


def test_no_ordering(figure):
    # We have no ordering
    with pytest.raises(TypeError):
        figure < figure
