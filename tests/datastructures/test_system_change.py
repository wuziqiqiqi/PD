from collections import Counter
import pytest
from ase.build import bulk
from cleases.jsonio import read_json
from cleases.datastructures import SystemChange


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


def test_default_name():
    s = SystemChange(0, "A", "B")
    assert s.name == ""
    s = SystemChange(0, "A", "B", "test")
    assert s.name == "test"


def test_step_order():
    """Test that system change is un-orderable"""
    change1 = SystemChange(0, "A", "B")
    change2 = SystemChange(1, "B", "A")
    with pytest.raises(TypeError):
        change1 < change2
    with pytest.raises(TypeError):
        change1 > change2


def test_eq():
    change1 = SystemChange(-1, "A", "B", "C")
    change2 = SystemChange(-1, "A", "B", "asdf")
    # name shouldn't be checked in equality
    assert change1 == change2
    change3 = SystemChange(0, "A", "C", "dummy")
    assert change1 != change3
    change4 = SystemChange(-1, "C", "B")
    assert change1 != change4
    assert change3 != change4


@pytest.mark.parametrize(
    "change",
    [
        SystemChange(-1, "", "", "A"),
        SystemChange(1, "A", "B", "A"),
        SystemChange(0, "C", "B", "A"),
    ],
)
def test_save_load(make_tempfile, change):
    file = make_tempfile("change.json")
    change.save(file)
    loaded = read_json(file)
    assert change == loaded
