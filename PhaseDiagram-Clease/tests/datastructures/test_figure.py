import pytest
from ase.build import bulk
import numpy as np
from clease.datastructures import FourVector, Figure
from clease.jsonio import read_json


@pytest.fixture
def make_random_figure(make_random_four_vector):
    def _make_random_figure(length):
        fvs = [make_random_four_vector() for _ in range(length)]
        return Figure(fvs)

    return _make_random_figure


@pytest.fixture
def figure():
    fv1 = FourVector(0, 0, 0, 0)
    fv2 = FourVector(0, 0, 0, 1)
    return Figure({fv1, fv2})


@pytest.fixture
def prim():
    return bulk("NaCl", crystalstructure="rocksalt", a=4.0)


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


def test_get_diameter(prim):
    def d(vec):
        """Helper function get the length of a vector"""
        return np.linalg.norm(vec)

    x, y, z = prim.get_cell()
    # Some safety checks that prim is as we expect
    assert d(x) == pytest.approx(d(y))
    assert len(prim) == 2

    # Pre-make a couple of FourVectors
    fv1 = FourVector(0, 0, 0, 0)
    fv2 = FourVector(0, 0, 0, 1)
    fv3 = FourVector(1, 0, 0, 0)

    # Binary should have the diameter the distance between the atoms
    dist = np.linalg.norm(prim[0].position - prim[1].position)
    assert Figure([fv1, fv2]).get_diameter(prim) == pytest.approx(dist)
    # 1 shift along the x vector
    assert Figure([fv1, fv3]).get_diameter(prim) == pytest.approx(d(x))
    fig = Figure([fv1, FourVector(1, 1, 0, 0)])
    assert fig.get_diameter(prim) == pytest.approx(d(x + y))
    fig = Figure([fv1, FourVector(3, 4, 1, 0)])
    assert fig.get_diameter(prim) == pytest.approx(d(3 * x + 4 * y + z))

    # Compare against some pre-calculated values for 3-body figures
    fig = Figure([fv1, fv2, fv3])
    assert fig.get_diameter(prim) == pytest.approx(4.0)

    fig = Figure([fv1, fv3, FourVector(3, 4, 1, 0)])
    assert fig.get_diameter(prim) == pytest.approx(23.776739333502675)

    # Square surrounding (0, 0) in the (x, y) plane.
    fig = Figure(
        [
            FourVector(-1, 0, 0, 0),
            FourVector(1, 0, 0, 0),
            FourVector(0, -1, 0, 0),
            FourVector(0, 1, 0, 0),
        ]
    )
    assert fig.get_diameter(prim) == pytest.approx(d(x) * 2)
    assert fig.get_diameter(prim) == pytest.approx(d(y) * 2)


def test_save_load(make_random_figure, make_tempfile):
    file = make_tempfile("figure.json")
    for i in range(25):
        figure = make_random_figure(i)
        figure.save(file)
        loaded = read_json(file)
        assert isinstance(loaded, Figure)
        assert figure == loaded
        assert figure is not loaded
        for fv in loaded.components:
            assert isinstance(fv, FourVector)
