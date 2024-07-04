import pytest
import numpy as np
from ase.build import bulk
from ase.db import connect
from clease.calculator import CleaseCacheCalculator
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture
def atoms():
    return bulk("Au")


def test_cache(atoms):
    calc = CleaseCacheCalculator(energy=1.2, forces=np.array([1, 2, 3]), foo="bar")
    atoms.calc = calc

    assert atoms.get_potential_energy() == pytest.approx(1.2)
    assert np.allclose(atoms.get_forces(), [1, 2, 3])
    assert calc.get_property("foo") == "bar"


def test_str(atoms):
    calc = CleaseCacheCalculator(energy=1.2, forces=np.array([1, 2, 3]))

    s = str(calc)
    assert "energy=1.2" in s
    assert "forces=..." in s


def test_save_load_to_db(db_name, atoms):
    con = connect(db_name)
    calc = CleaseCacheCalculator(energy=1.2, forces=np.array([[0, 0, 3.3]]), dummy=np.array([1]))
    atoms.calc = calc

    assert calc is atoms.calc

    con.write(atoms)

    row = con.get(id=1)
    # Verify we can recover the attributes
    assert row.energy == calc.get_potential_energy()
    assert row.fmax == pytest.approx(3.3)

    # ASE will always make the calculator on the loaded a SinglePoint Calculator
    # Unfortunately, ASE will not retain the dummy array we put in.
    # But we still check that it doesn't fail the write.
    loaded = row.toatoms()
    assert isinstance(loaded.calc, SinglePointCalculator)
    assert loaded.calc.name == calc.name
