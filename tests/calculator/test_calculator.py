import pytest
import numpy as np
from clease.settings import Concentration, CEBulk
from clease.corr_func import CorrFunction
from clease.calculator import Clease, attach_calculator
from clease.calculator.clease import UnitializedCEError
from clease.datastructures import SystemChange
from clease import tools
from clease_cxx import has_parallel


@pytest.fixture
def settings(db_name):
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    settings = CEBulk(
        db_name=db_name,
        concentration=conc,
        crystalstructure="fcc",
        a=4.0,
        max_cluster_dia=[4.0],
        size=[1, 1, 1],
    )
    return settings


@pytest.fixture
def atoms(settings):
    atoms_ = settings.prim_cell.copy() * (3, 3, 3)
    # Insert a few different symbols
    atoms_.symbols = "Au"
    atoms_.symbols[:10] = "Cu"
    return atoms_


@pytest.fixture
def example_system(atoms, settings):
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci["c0"] = -1.0
    eci["c2_d0000_0_00"] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    return atoms


@pytest.fixture
def simple_change(atoms):
    assert atoms.symbols[0] == "Cu"
    assert atoms.symbols[11] == "Au"
    changes = [
        SystemChange(0, "Cu", "Au", "dummy"),
        SystemChange(11, "Au", "Cu", "dummy"),
    ]
    return changes


def test_apply_system_changes(example_system, simple_change):
    atoms = example_system
    calc = atoms.calc

    E0 = calc.get_energy()
    # Test we can apply changes
    calc.apply_system_changes(simple_change)
    for change in simple_change:
        assert atoms.symbols[change.index] == change.new_symb
        assert atoms.symbols[change.index] != change.old_symb
    E1 = calc.get_energy()
    print(E0, E1)
    assert E0 != pytest.approx(E1)

    # Undoing it again
    calc.undo_system_changes()
    for change in simple_change:
        assert atoms.symbols[change.index] == change.old_symb
        assert atoms.symbols[change.index] != change.new_symb
    E2 = calc.get_energy()
    assert E0 == pytest.approx(E2)


def test_keep_changes(example_system, simple_change):
    atoms = example_system
    calc = atoms.calc

    calc.apply_system_changes(simple_change)
    atoms_cpy = atoms.copy()

    E0 = calc.get_energy()
    calc.keep_system_changes()
    # Undo should now do nothing after keeping changes
    calc.undo_system_changes()
    assert all(atoms.symbols == atoms_cpy.symbols)
    E1 = calc.get_energy()
    assert pytest.approx(E0) == E1


def test_attach_calculator(atoms, settings):
    eci = {"c0": 0.5, "c1_0": 0.3}
    new_atoms = attach_calculator(settings, atoms, eci)
    cf1 = new_atoms.calc.get_cf()

    new_atoms.calc.updater.calculate_cf_from_scratch(atoms, eci.keys())
    cf2 = new_atoms.calc.get_cf()
    assert cf1 == pytest.approx(cf2)


@pytest.mark.parametrize(
    "P",
    [
        [[9, -9, 0], [0, 9, -9], [3, 3, 3]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[3, 0, 0], [0, 2, 0], [0, 0, 4]],
    ],
)
@pytest.mark.parametrize("rattle", [0.01, 0.001, 0.0001])
def test_attach_calc_bad_atoms(settings, P, rattle):
    supercell = tools.make_supercell(settings.prim_cell, P)
    supercell.rattle(rattle)
    with pytest.raises(ValueError):
        attach_calculator(settings, supercell)


def test_attach_calc_default_threads(settings):
    """Number of threads without specifying anything should be 1,
    even when not compiled with OpenMP."""
    atoms = settings.prim_cell.copy()
    atoms = attach_calculator(settings, atoms)
    assert atoms.calc.get_num_threads() == 1


def test_calc_wrong_threads(settings):
    """Number of threads without specifying anything should be 1,
    even when not compiled with OpenMP."""
    atoms = settings.prim_cell.copy()
    atoms = attach_calculator(settings, atoms)
    assert atoms.calc.get_num_threads() == 1

    if not has_parallel():
        # This only fails if not compiled with OpenMP
        with pytest.raises(ValueError):
            atoms.calc.set_num_threads(2)
    else:
        atoms.calc.set_num_threads(2)
    # Fractional threads
    for n in [1.2, 1.5, 2.7, 3.9, None]:
        with pytest.raises(TypeError):
            atoms.calc.set_num_threads(n)
    with pytest.raises(ValueError):
        atoms.calc.set_num_threads(0)


def test_calc_no_initialization(settings):
    """No atoms object has been attached yet"""
    eci = {"c0": 0.0}
    calc = Clease(settings, eci)
    assert calc.updater is None
    with pytest.raises(UnitializedCEError):
        calc.set_num_threads(1)

    with pytest.raises(UnitializedCEError):
        calc.get_num_threads()
