from collections import Counter
import pytest
from ase.build import bulk
from clease.datastructures import SystemChange


@pytest.fixture
def atoms():
    return bulk("Al", crystalstructure="sc", a=4.0)


def test_system_change(atoms):
    supercell = atoms * (4, 4, 4)
    natoms = len(supercell)

    sym_count = Counter(supercell.symbols)
    assert len(sym_count) == 1
    # Get the symbol
    sym = supercell.symbols[0]
    assert sym == "Al"
    new_sym = "Zn"
    index = 4
    name = "test_change"

    # Create a system change
    change = SystemChange(index, sym, new_sym, name)
    assert change.index == index
    assert change.new_symb == new_sym
    assert change.old_symb == sym
    assert change.name == name

    change.apply_change(supercell)
    # Check the the supercell was correctly mutated
    sym_count = Counter(supercell.symbols)
    assert len(sym_count) == 2  # We should have 'Al' and 'Zn' now
    assert sym_count[new_sym] == 1
    # Total number of atoms should not have changed
    assert sym_count[sym] + sym_count[new_sym] == natoms
    # Check we changed the correct symbol
    assert supercell.symbols[index] == new_sym

    # Undo the changes
    change.undo_change(supercell)
    sym_count = Counter(supercell.symbols)
    assert len(sym_count) == 1
    assert sym_count[sym] == natoms
    assert supercell.symbols[index] == sym
