from collections import Counter
from ase import Atoms
from ase.build import bulk
from ase.spacegroup import crystal
import pytest
import numpy as np
from clease.tools import make_supercell
from clease.jsonio import read_json
from clease.datastructures import FourVector, construct_four_vectors


def make_scaled_crd_close_to_one(tol: float = 1e-12):
    """
    Helper function to construct an atoms object with one atom close
    to an edge with scaled coordinate equal to 1
    """
    return Atoms(
        symbols=["Au"],
        positions=[[0.0, 0.0, 1.0 - tol]],
        cell=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pbc=True,
    )


def make_scaled_crd_close_to_zero(tol: float = 1e-12):
    """
    Helper function to construct an atoms object with one atom close
    to an edge with scaled coordinate equal to 1
    """
    return Atoms(
        symbols=["Au"],
        positions=[[0.0, 0.0, -tol]],
        cell=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pbc=True,
    )


def make_wrapped_nacl():
    atoms = bulk("NaCl", a=4.0, crystalstructure="rocksalt", cubic=False)
    atoms.wrap()
    return atoms


@pytest.mark.parametrize("inputs", [(0.5, 1, 1, 1), (1, 0.5, 1, 1), (1, 1, 0.5, 1), (1, 1, 1, 0.5)])
def test_invalid_fv(inputs):
    with pytest.raises(TypeError):
        FourVector(*inputs)._validate()


def test_int_np_int_hash():
    """Test that python int and np.int64/np.integer
    behaves identical
    """
    a = 4
    a_np = np.int_(4)
    assert a == a_np
    assert hash(a) == hash(a_np)
    assert type(a) != type(a_np)

    fv1 = FourVector(0, 0, 0, 0)
    fv1._validate()
    for v in fv1.to_tuple():
        # Verify the elements are regular int, and not numpy int
        assert isinstance(v, int)
        assert not isinstance(v, np.integer)

    arr = np.array([0, 0, 0, 0], dtype=np.int_)
    fv2 = FourVector(*arr)
    for v in fv2.to_tuple():
        # Verify each element is numpy integer type
        assert isinstance(v, np.integer)
        assert isinstance(v, np.int_)
        assert not isinstance(v, int)
    assert fv1 == fv2
    assert hash(fv1) == hash(fv2)

    # Verify that a set of the regular python int FourVector
    # and the np.integer becomes just one element.
    assert len(set([fv1, fv2])) == 1


@pytest.mark.parametrize(
    "test",
    [
        # Test various trivial repetitions of a single atom cell
        {
            "prim": bulk("Na", cubic=False),
            "atoms": bulk("Na", cubic=False),
            "expect": [FourVector(0, 0, 0, 0)],
        },
        {
            "prim": bulk("Na", cubic=False),
            "atoms": bulk("Na", cubic=False) * (1, 1, 2),
            "expect": [FourVector(0, 0, 0, 0), FourVector(0, 0, 1, 0)],
        },
        {
            "prim": bulk("Na", cubic=False),
            "atoms": bulk("Na", cubic=False) * (1, 2, 1),
            "expect": [FourVector(0, 0, 0, 0), FourVector(0, 1, 0, 0)],
        },
        {
            "prim": bulk("Na", cubic=False),
            "atoms": bulk("Na", cubic=False) * (2, 1, 1),
            "expect": [FourVector(0, 0, 0, 0), FourVector(1, 0, 0, 0)],
        },
        # Test cubic version of FCC
        {
            "prim": bulk("Al", cubic=False),
            "atoms": bulk("Al", cubic=True),
            "expect": [
                FourVector(0, 0, 0, 0),
                FourVector(1, 0, 0, 0),
                FourVector(0, 1, 0, 0),
                FourVector(0, 0, 1, 0),
            ],
        },
        # Test cubic supercell
        {
            "prim": bulk("Al", cubic=False),
            "atoms": bulk("Al", cubic=True) * (2, 1, 1),
            "expect": [
                FourVector(0, 0, 0, 0),
                FourVector(1, 0, 0, 0),
                FourVector(0, 1, 0, 0),
                FourVector(0, 0, 1, 0),
                FourVector(-1, 1, 1, 0),
                FourVector(0, 1, 1, 0),
                FourVector(-1, 2, 1, 0),
                FourVector(-1, 1, 2, 0),
            ],
        },
        # Test rocksalt contains one atom outisde the box. Thus, one of the atoms
        # belongs to cell with (-1, 0, 0)
        {
            "prim": make_wrapped_nacl(),
            "atoms": bulk("NaCl", a=4.0, crystalstructure="rocksalt", cubic=False),
            "expect": [FourVector(0, 0, 0, 0), FourVector(-1, 0, 0, 1)],
        },
        {
            "prim": make_wrapped_nacl(),
            "atoms": make_wrapped_nacl(),
            "expect": [FourVector(0, 0, 0, 0), FourVector(0, 0, 0, 1)],
        },
        # Test atoms object with an atom relatively close to an edge with scaled
        # coordinate 1.0
        {
            "prim": make_scaled_crd_close_to_one(1e-3),
            "atoms": make_scaled_crd_close_to_one(1e-3),
            "expect": [FourVector(0, 0, 0, 0)],
        },
        # Test atoms object with an atom very close to an edge with scaled
        # coordinate 1.0
        {
            "prim": make_scaled_crd_close_to_one(1e-12),
            "atoms": make_scaled_crd_close_to_one(1e-12),
            "expect": [FourVector(0, 0, 0, 0)],
        },
        # Test when the atoms in an atoms object has ended up slig
        {
            "prim": make_scaled_crd_close_to_zero(-1e-12),
            "atoms": make_scaled_crd_close_to_zero(1e-12),
            "expect": [FourVector(0, 0, 0, 0)],
        },
    ],
)
def test_four_vector(test):
    for fv in test["expect"]:
        fv._validate()
    assert construct_four_vectors(test["prim"], test["atoms"]) == test["expect"]


def test_random_spacegroup_four_vector(default_seed):
    """
    Tests nessecary conditions for four vectors for random atoms object

    1. The number of sites matches parent atoms
    2. All 4-vectors are unique
    3. Number of atoms in all "sub-cells" matches the number in primitive
    4. Build the atoms object from the 4-vector
    """
    rng = np.random.default_rng(42)
    num = 20
    for i in range(num):
        sp_group = int(rng.integers(1, 230))
        num_atoms = rng.integers(2, 6)
        basis = rng.random((num_atoms, 3))

        cell = rng.random((3, 3)) * 16.0 - 8.0
        prim = crystal(symbols=["Al"] * num_atoms, basis=basis, spacegroup=sp_group, cell=cell)

        # Transformation matrix
        P = rng.integers(1, 3, size=(3, 3))
        # Set elements below diagonal to zero (avoid issues with singular)
        P[1, 0] = P[2, 0] = P[2, 1] = 0

        supercell = make_supercell(prim, P)

        four_vecs = construct_four_vectors(prim, supercell)
        for fv in four_vecs:
            fv._validate()

        # Correct number of atoms
        assert len(four_vecs) == len(supercell)

        # Unique four vectors
        assert len(set(four_vecs)) == len(four_vecs)

        # Correct number in each sub-cell
        num_in_each_sublattice = Counter(fv.sublattice for fv in four_vecs)
        detP = np.linalg.det(P)
        assert all(abs(v - detP) < 1e-12 for v in num_in_each_sublattice.values())

        # Make sure that we have the correct number of sublattices
        assert len(num_in_each_sublattice) == len(prim)

        # Re-build the atoms object
        positions = [vec.to_cartesian(prim) for vec in four_vecs]
        assert np.allclose(positions, supercell.get_positions())


@pytest.mark.parametrize(
    "test",
    [
        {
            "fv": FourVector(0, 0, 0, 0),
            "prim": bulk("Al"),
            "expect": np.array([0.0, 0.0, 0.0]),
        },
        {
            "fv": FourVector(0, 0, 0, 0),
            "prim": make_wrapped_nacl(),
            "expect": np.array([0.0, 0.0, 0.0]),
        },
        {
            "fv": FourVector(0, 0, 0, 1),
            "prim": make_wrapped_nacl(),
            "expect": np.array([0.5, 0.5, 0.5]),
        },
        {
            "fv": FourVector(1, 0, 0, 1),
            "prim": make_wrapped_nacl(),
            "expect": np.array([1.5, 0.5, 0.5]),
        },
    ],
)
def test_to_scaled_src(test):
    assert np.allclose(test["fv"].to_scaled(test["prim"]), test["expect"])


def test_shift_xyz():
    fv1 = FourVector(0, 0, 0, 0)
    fv2 = FourVector(1, 0, 0, 1)

    res1 = fv1.shift_xyz(fv2)
    # We created new FourVector instances
    assert res1 is not fv1
    assert res1 is not fv2
    res2 = fv2.shift_xyz(fv1)

    # Ordering on the sublattice matters.
    assert res1 == FourVector(1, 0, 0, 0)
    assert res2 == FourVector(1, 0, 0, 1)


def test_shift_notimplemented():
    fv = FourVector(0, 0, 0, 0)
    with pytest.raises(NotImplementedError):
        fv.shift_xyz((0, 0, 0, 1))
    with pytest.raises(NotImplementedError):
        fv.shift_xyz([0, 0, 0, 1])


def test_save_load(make_random_four_vector, make_tempfile):
    file = make_tempfile("four_vector.json")
    for _ in range(20):
        fv = make_random_four_vector(min=-20, max=20)
        fv.save(file)
        loaded = read_json(file)
        assert isinstance(loaded, FourVector)
        assert fv == loaded
        assert fv is not loaded


def test_xyz_array(make_random_four_vector):
    for _ in range(20):
        fv = make_random_four_vector(min=-20, max=20)
        xyz = fv.xyz_array
        assert isinstance(xyz, np.ndarray)
        assert xyz.shape == (3,)
        assert np.issubdtype(xyz.dtype, np.integer)
