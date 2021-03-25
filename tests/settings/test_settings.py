import pytest
from ase.build import bulk
from clease.settings import CEBulk, Concentration
from clease.calculator import Clease
from clease.tools import wrap_and_sort_by_position


@pytest.fixture
def dummy_eci():
    return {'c0': 0.0}


@pytest.fixture
def settings_and_atoms(db_name):
    """Create some bulk settings, and a matching AuCu atoms bulk cell"""
    a = 3.8  # Lattice parameter
    basis_elements = [["Au", "Cu"]]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure="fcc",
                      a=a,
                      size=[3, 3, 3],
                      concentration=concentration,
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[5.0, 5.0])

    atoms = bulk("Au", crystalstructure="fcc", a=a)
    atoms = atoms * (3, 3, 3)
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"

    atoms = wrap_and_sort_by_position(atoms)
    return settings, atoms


def test_get_figures_settings(settings_and_atoms, dummy_eci):
    """Regression test, see issue #263.
    
    After getting figures from the settings' cluster manager,
    attaching a calculator would result in a RuntimeError.
    """
    settings, atoms = settings_and_atoms
    eci = dummy_eci

    # This part crashed in #263
    settings.cluster_mng.get_figures()
    calc = Clease(settings, eci)
    atoms.calc = calc
    assert atoms.calc is not None
    assert isinstance(atoms.calc, Clease)


if __name__ == '__main__':
    pytest.main([__file__])
