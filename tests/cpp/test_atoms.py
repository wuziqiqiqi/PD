from clease_cxx import CppAtoms

import copy
import numpy as np
from ase.build import bulk
from cleases.datastructures.four_vector import construct_four_vectors, FourVector
from cleases.datastructures.system_changes import SystemChange
import pytest


@pytest.fixture
def prim():
    ats = bulk("NaCl", crystalstructure="rocksalt", a=4.0)
    ats.wrap()
    return ats


@pytest.fixture
def atoms(prim):
    return prim * (3, 3, 3)


@pytest.fixture
def cpp_atoms(atoms, four_vectors):
    cpp_atoms = CppAtoms(atoms, four_vectors)
    return cpp_atoms


@pytest.fixture
def four_vectors(prim, atoms):
    return construct_four_vectors(prim, atoms)


def test_initialization(cpp_atoms, atoms, four_vectors):
    res = cpp_atoms.get_four_vectors()
    assert res == four_vectors
    for fv in res:
        assert isinstance(fv, FourVector)

    nums = cpp_atoms.get_numbers()
    assert np.all(atoms.numbers == nums)

    syms = cpp_atoms.get_symbols()
    assert np.all(atoms.symbols == syms)


def test_access_fv(prim, atoms):
    """During development, there was an issue that accessing the four vectors
    (due to a stray DECREF call) would segfault after constructing the
    CppAtoms if an error was raised.
    Ensure we can raise after constructing CppAtoms.
    """
    four_vectors = construct_four_vectors(prim, atoms)
    cpp_atoms = CppAtoms(atoms, four_vectors)
    for fv in four_vectors:
        pass


def test_Ns(cpp_atoms, four_vectors, prim):
    Ns_expect = max(fv.sublattice for fv in four_vectors) + 1
    assert cpp_atoms.Ns == Ns_expect
    assert cpp_atoms.Ns == len(prim)


@pytest.mark.parametrize("name, attr", [("Nx", "ix"), ("Ny", "iy"), ("Nz", "iz")])
def test_max_4vec_values(cpp_atoms, four_vectors, name, attr):
    expect = max(getattr(fv, attr) for fv in four_vectors) + 1
    assert getattr(cpp_atoms, name) == expect


def test_get_1d_index(cpp_atoms, four_vectors):
    indices = [cpp_atoms.get_1d_index(fv) for fv in four_vectors]
    assert len(set(indices)) == len(indices)
    # Ensure we get all the values from 0 to len(four_vectors) - 1
    # when we order them.
    assert sorted(indices) == list(range(len(four_vectors)))


def test_apply_change(cpp_atoms):
    syms = cpp_atoms.get_symbols()
    syms_cpy = copy.deepcopy(syms)

    change = SystemChange(0, "Na", "Au", "dummy")
    cpp_atoms.apply_change(change)

    assert syms == syms_cpy
    syms_new = cpp_atoms.get_symbols()

    assert syms_new != syms
    assert syms[0] == "Na"
    assert syms_new[0] == "Au"

    cpp_atoms.undo_change(change)
    syms_undone = cpp_atoms.get_symbols()
    assert syms_undone[0] == "Na"
    assert syms_new[0] == "Au"
