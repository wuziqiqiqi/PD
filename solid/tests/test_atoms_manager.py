import pytest
from ase.build import bulk
from clease.settings.atoms_manager import AtomsManager


def test_binary():
    atoms = bulk("Au") * (3, 3, 3)

    # Tag even indices with 0 and odd indices with 1
    for atom in atoms:
        atom.tag = atom.index % 2

    manager = AtomsManager(atoms)
    ind_by_tag = manager.index_by_tag()
    assert all(map(lambda x: x % 2 == 0, ind_by_tag[0]))
    assert all(map(lambda x: x % 2 == 1, ind_by_tag[1]))


def test_group_by_symbol_single():
    atoms = bulk("Au") * (3, 3, 3)

    for atom in atoms:
        if atom.index % 3 == 1:
            atom.symbol = "Cu"
        elif atom.index % 3 == 2:
            atom.symbol = "X"

    manager = AtomsManager(atoms)
    ind_by_symbol = manager.index_by_symbol(["Au", "Cu", "X"])

    for i, items in enumerate(ind_by_symbol):
        assert all(map(lambda x: x % 3 == i, items))


def test_group_by_symbol_grouped():
    atoms = bulk("Au") * (3, 4, 5)

    for atom in atoms:
        if atom.index % 4 == 1:
            atom.symbol = "Cu"
        elif atom.index % 4 == 2:
            atom.symbol = "X"
        elif atom.index % 4 == 3:
            atom.symbol = "Ag"

    manager = AtomsManager(atoms)
    ind_by_symbol = manager.index_by_symbol(["Au", ["Cu", "X"], "Ag"])

    assert all(map(lambda x: x % 4 == 0, ind_by_symbol[0]))
    assert all(map(lambda x: x % 4 == 1 or x % 4 == 2, ind_by_symbol[1]))
    assert all(map(lambda x: x % 4 == 3, ind_by_symbol[2]))
    assert sorted(manager.unique_elements()) == ["Ag", "Au", "Cu", "X"]


def test_background_indices():
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=4.0)
    atoms = atoms * (5, 5, 5)

    # Chlorine sites are background indices
    basis_elements = [["Na", "X"], ["Cl"]]
    manager = AtomsManager(atoms)
    bkg_indices = manager.single_element_sites(basis_elements)

    cl_sites = [atom.index for atom in atoms if atom.symbol == "Cl"]
    assert sorted(bkg_indices) == sorted(cl_sites)

    # Extract unique elements
    unique_elem = manager.unique_elements()
    assert sorted(unique_elem) == ["Cl", "Na"]

    # Try unique elements without background
    unique_elem = manager.unique_elements(ignore=["Cl"])
    assert sorted(unique_elem) == ["Na"]


@pytest.mark.parametrize(
    "atoms1,atoms2",
    [
        (bulk("Au") * (3, 3, 3), bulk("Au") * (4, 4, 4)),
        (bulk("Au") * (3, 3, 3), bulk("NaCl", crystalstructure="rocksalt", a=4.0)),
        (
            bulk("NaCl", crystalstructure="rocksalt", a=4.0),
            bulk("NaCl", crystalstructure="rocksalt", a=4.0) * (2, 2, 2),
        ),
    ],
)
def test_equality(atoms1, atoms2):
    m1 = AtomsManager(atoms1)
    m2 = AtomsManager(atoms2)

    assert m1 != m2
    assert m1 != atoms1
    assert m1 == AtomsManager(atoms1)
    assert m2 == AtomsManager(atoms2)
    assert m2 != "some_string"

    m2.atoms = atoms1
    assert m1 == m2
