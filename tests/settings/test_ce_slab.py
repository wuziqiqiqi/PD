import pytest
import numpy as np
import ase
from ase.build import bulk
from clease.settings.settings_slab import (
    get_prim_slab_cell,
    add_vacuum_layers,
    remove_vacuum_layers,
    CESlab,
)
from clease.settings import settings_from_json, Concentration

# Skip entire module if we are not above ASE 3.19
pytestmark = pytest.mark.skipif(ase.__version__ < "3.19", reason="CESlab requires ASE > 3.19")


@pytest.mark.parametrize(
    "test",
    [
        {
            "cell": bulk("Al", a=4.05, cubic=True),
            "miller": [1, 1, 1],
            "expect_dist": 4.05 / np.sqrt(3.0),
        },
        {
            "cell": bulk("Al", a=4.05, cubic=True),
            "miller": [1, 1, 0],
            "expect_dist": 4.05 / np.sqrt(2.0),
        },
        {
            "cell": bulk("Fe", a=4.05, cubic=True),
            "miller": [1, 1, 1],
            "expect_dist": 4.05 / np.sqrt(3.0),
        },
        {
            "cell": bulk("Fe", a=4.05, cubic=True),
            "miller": [1, 1, 0],
            "expect_dist": 4.05 / np.sqrt(2.0),
        },
    ],
)
def test_prim_cell_construction(test):
    prim = get_prim_slab_cell(test["cell"], test["miller"])
    dist = prim.get_cell()[2, 2]
    assert dist == pytest.approx(test["expect_dist"])


def test_add_vacuum_layers():
    atoms = bulk("Al", a=4.05, cubic=True)
    prim = get_prim_slab_cell(atoms, [1, 1, 1])
    z_prim = prim.cell[2, 2]
    atoms = prim * (1, 1, 3)
    z_orig = atoms.cell[2, 2]
    atoms = add_vacuum_layers(atoms, prim, 10.0)
    new_z = atoms.cell[2, 2]
    z_vac = int(-(-10.0 // z_prim)) * z_prim
    assert z_vac > 10
    assert new_z == pytest.approx(z_orig + z_vac)

    tol = 1e-6
    for atom in atoms:
        if atom.position[2] > z_orig - tol:
            assert atom.symbol == "X"
        else:
            assert atom.symbol == "Al"


def test_load(db_name, make_tempfile):
    atoms = bulk("Al", a=4.05, cubic=True)
    conc = Concentration(basis_elements=[["Al", "X"]])
    settings = CESlab(atoms, (1, 1, 1), conc, db_name=db_name)

    backup_file = make_tempfile("test_save_ceslab.json")
    settings.save(backup_file)

    settings2 = settings_from_json(backup_file)

    assert settings.atoms == settings2.atoms
    assert settings.size == settings2.size
    assert settings.concentration == settings2.concentration


def test_remove_vacuum():
    unit_cell = bulk("Au", crystalstructure="fcc", cubic=True)
    prim = get_prim_slab_cell(unit_cell, [1, 1, 1])
    atoms = prim * (3, 3, 3)
    slab = add_vacuum_layers(atoms.copy(), prim, thickness=20)
    recovered = remove_vacuum_layers(slab)
    assert atoms == recovered
