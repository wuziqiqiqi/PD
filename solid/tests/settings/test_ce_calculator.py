"""Unit tests for the Clease calculator."""
import os
import time
from pathlib import Path
import json
import pytest
import numpy as np
from random import randint, choice
from ase.build import bulk
from ase.spacegroup import crystal

from clease.tools import wrap_and_sort_by_position
from clease.datastructures import SystemChange, TransMatrix
from clease.settings import CEBulk, CECrystal
from clease.corr_func import CorrFunction
from clease.settings import Concentration
from clease.calculator import Clease, attach_calculator, get_ce_energy

# Only change this to override the reference trans matrices
UPDATE_TRANS_MATRIX = False

ALL_BASIS_FUNCTIONS = ["polynomial", "binary_linear", "trigonometric"]


@pytest.fixture
def save_trans_matrix(references_path):
    def _save_trans_matrix(filename, tm):
        with open(references_path / filename, "w") as file:
            json.dump(tm, file)

    return _save_trans_matrix


@pytest.fixture
def load_trans_matrix(references_path):
    def _load_trans_matrix(filename):
        with open(references_path / filename) as file:
            loaded = json.load(file)
        # Convert keys to integers, not strings
        tm = TransMatrix([{int(k): v for k, v in dct.items()} for dct in loaded])
        return tm

    return _load_trans_matrix


def generate_ex_eci(settings):
    """Return dummy ECIs. All are set to -0.001."""
    cf = CorrFunction(settings)
    cf = cf.get_cf(settings.atoms)
    eci = {key: -0.001 for key in cf}
    return eci


def get_binary(db_name):
    """Return a simple binary test structure."""
    basis_elements = [["Au", "Cu"]]
    concentration = Concentration(basis_elements=basis_elements)
    bc_settings = CEBulk(
        crystalstructure="fcc",
        a=4.05,
        size=[3, 3, 3],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[5.0, 5.0],
    )

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms * (3, 3, 3)
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"
    atoms = wrap_and_sort_by_position(atoms)
    bc_settings.set_active_template(atoms=atoms)
    return bc_settings, atoms


def get_ternary(db_name):
    """Return a ternary test structure."""
    basis_elements = [["Au", "Cu", "Zn"]]
    concentration = Concentration(basis_elements=basis_elements)
    bc_settings = CEBulk(
        crystalstructure="fcc",
        a=4.05,
        size=[3, 3, 3],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[5.0, 5.0],
    )

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms * (3, 3, 3)
    for i in range(2):
        atoms[3 * i].symbol = "Au"
        atoms[3 * i + 1].symbol = "Cu"
        atoms[3 * i + 2].symbol = "Zn"
    atoms = wrap_and_sort_by_position(atoms)
    bc_settings.set_active_template(atoms=atoms)
    return bc_settings, atoms


def get_rocksalt(db_name, bf_type="trigonometric"):
    """Test rocksalt where passed atoms with background_atoms."""
    basis_elements = [["Li", "X", "V"], ["O"]]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        crystalstructure="rocksalt",
        a=4.05,
        size=[3, 3, 3],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[7.0, 7.0],
    )
    settings.basis_func_type = bf_type
    atoms = bulk("LiO", crystalstructure="rocksalt", a=4.05)
    atoms = atoms * (3, 3, 3)
    Li_indx = [a.index for a in atoms if a.symbol == "Li"]
    for i in range(18):
        if i < 9:
            atoms[Li_indx[i]].symbol = "V"
        else:
            atoms[Li_indx[i]].symbol = "X"
    atoms = wrap_and_sort_by_position(atoms)
    settings.set_active_template(atoms=atoms)
    return settings, atoms


def rocksalt_with_self_interaction(size, db_name, bf_type="trigonometric"):
    basis_elements = [["Li", "Mn", "X"], ["O", "X"]]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        crystalstructure="rocksalt",
        a=4.05,
        size=size,
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[7.0, 4.0],
    )
    settings.basis_func_type = bf_type
    atoms = settings.atoms.copy()
    settings.set_active_template(atoms=atoms)
    return settings, atoms


def get_spacegroup(db_name, bf_type="trigonometric"):
    """Test rocksalt where passed atoms."""
    basis = [
        (0.0, 0.0, 0.0),
        (0.3894, 0.1405, 0.0),
        (0.201, 0.3461, 0.5),
        (0.2244, 0.3821, 0.0),
    ]
    spacegroup = 55
    cellpar = [6.25, 7.4, 3.83, 90, 90, 90]
    size = [2, 2, 2]
    basis_elements = [["O", "X"], ["O", "X"], ["O", "X"], ["Ta"]]
    grouped_basis = [[0, 1, 2], [3]]
    concentration = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    settings = CECrystal(
        basis=basis,
        spacegroup=spacegroup,
        cellpar=cellpar,
        size=size,
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[4.0, 4.0],
        include_background_atoms=True,
    )
    settings.basis_func_type = bf_type

    atoms = crystal(
        symbols=["O", "X", "O", "Ta"],
        basis=basis,
        spacegroup=spacegroup,
        cell=None,
        cellpar=cellpar,
        ab_normal=(0, 0, 1),
        primitive_cell=True,
        size=size,
    )
    atoms = wrap_and_sort_by_position(atoms)
    settings.set_active_template(atoms=atoms)
    return settings, atoms


def do_test_update_correlation_functions(settings, atoms, n_trial_configs=20, fixed=()):
    """Perform swaps and check that the correlation functions match.

    The comparison is done by check that each CF in the Clease
    calculator is the same as the ones obtained by direct calculation.
    """
    settings.set_active_template(atoms=atoms)
    cf = CorrFunction(settings)

    eci = generate_ex_eci(settings)
    calc = Clease(settings, eci=eci)
    atoms.calc = calc

    timings = []
    for _ in range(n_trial_configs):
        indx1 = randint(0, len(atoms) - 1)
        symb1 = atoms[indx1].symbol
        while symb1 in fixed:
            indx1 = randint(0, len(atoms) - 1)
            symb1 = atoms[indx1].symbol
        symb2 = symb1
        while symb2 == symb1 or symb2 in fixed:
            indx2 = randint(0, len(atoms) - 1)
            symb2 = atoms[indx2].symbol

        atoms[indx1].symbol = symb2
        atoms[indx2].symbol = symb1

        # The calculator should update its correlation functions
        # when the energy is computed
        start = time.perf_counter()
        calc.update_cf()
        en_update = calc.get_energy()
        timings.append(time.perf_counter() - start)
        brute_force_cf = cf.get_cf_by_names(atoms, calc.cf_names)
        # Get the CF's with the "smart" update method first
        calc_cf = calc.get_cf()
        assert pytest.approx(calc_cf) == brute_force_cf

        # Now verify we get the same CF's calculating with brute force from the calculator.
        calc_brute_force = calc.calculate_cf_from_scratch()
        assert pytest.approx(calc_cf) == calc_brute_force
        assert pytest.approx(en_update) == calc.calculate_energy_from_scratch()

    print(np.mean(timings))


def do_test_insert_element(settings, atoms, n_trial_configs=20):
    cf = CorrFunction(settings)
    eci = generate_ex_eci(settings)
    calc = Clease(settings, eci=eci)
    atoms.calc = calc
    elements = settings.unique_elements
    for _ in range(n_trial_configs):
        indx1 = randint(0, len(atoms) - 1)
        symb1 = atoms[indx1].symbol
        symb2 = symb1

        while symb2 == symb1:
            symb2 = choice(elements)
        atoms[indx1].symbol = symb2
        atoms.get_potential_energy()
        brute_force_cf = cf.get_cf_by_names(atoms, calc.cf_names)
        calc_cf = calc.get_cf()
        for k in calc_cf.keys():
            if k.startswith("c0") or k.startswith("c1"):
                continue
            assert abs(calc_cf[k] - brute_force_cf[k]) < 1e-6

        assert calc.calculate_cf_from_scratch() == pytest.approx(brute_force_cf)


def test_normfactors_no_self_interaction(db_name):
    settings, atoms = get_binary(db_name)

    eci = generate_ex_eci(settings)

    for cluster in settings.cluster_list:
        if cluster.name == "c0" or cluster.name == "c1":
            continue
        norm_factors = cluster.info["normalization_factor"]
        assert np.allclose(norm_factors, 1.0)


def test_indices_of_changed_symbols(db_name):
    settings, atoms = get_binary(db_name)
    eci = generate_ex_eci(settings)
    calc = Clease(settings, eci=eci)
    atoms.calc = calc

    changes = [2, 6]
    for ch in changes:
        if atoms[ch].symbol == "Au":
            atoms[ch].symbol = "Cu"
        else:
            atoms[ch].symbol = "Au"

    calc_changes = calc.indices_of_changed_atoms
    assert calc_changes == changes


def test_update_corr_func_binary(db_name, verify_clusters):
    print("binary")
    bin_settings, bin_atoms = get_binary(db_name)
    do_test_update_correlation_functions(bin_settings, bin_atoms, n_trial_configs=5)
    verify_clusters(bin_settings)


def test_update_corr_func_ternary(db_name, verify_clusters):
    print("ternary")
    tern_settings, tern_atoms = get_ternary(db_name)
    do_test_update_correlation_functions(tern_settings, tern_atoms, n_trial_configs=5)
    verify_clusters(tern_settings)


@pytest.mark.parametrize("basis_func_type", ALL_BASIS_FUNCTIONS)
def test_update_corr_func_rocksalt(db_name, basis_func_type):
    print("rocksalt")
    rs_settings, rs_atoms = get_rocksalt(db_name, bf_type=basis_func_type)
    assert rs_settings.cluster_list.assume_no_self_interactions
    do_test_update_correlation_functions(rs_settings, rs_atoms, n_trial_configs=5, fixed=["O"])


@pytest.mark.parametrize("basis_func_type", ALL_BASIS_FUNCTIONS)
def test_insert_element_rocksalt_1x1x1(db_name, verify_clusters, basis_func_type):
    print("rocksalt with self interaction 1x1x1")
    rs_settings, rs_atoms = rocksalt_with_self_interaction(
        [1, 1, 1], db_name, bf_type=basis_func_type
    )
    rs_settings.basis_func_type = basis_func_type
    # This test has self interactions
    assert not rs_settings.cluster_list.assume_no_self_interactions
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=5)
    verify_clusters(rs_settings)


@pytest.mark.parametrize("basis_func_type", ALL_BASIS_FUNCTIONS)
def test_insert_element_rocksalt_1x1x2(db_name, verify_clusters, basis_func_type):
    print("rocksalt with self interaction 1x1x2")
    rs_settings, rs_atoms = rocksalt_with_self_interaction(
        [1, 1, 2], db_name, bf_type=basis_func_type
    )
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=1)
    verify_clusters(rs_settings)


def test_insert_element_rocksalt_1x1x3(db_name):
    print("rocksalt with self interaction 1x1x3")
    rs_settings, rs_atoms = rocksalt_with_self_interaction([1, 1, 3], db_name)
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=10)


def test_insert_element_rocksalt_1x2x3(db_name):
    print("rocksalt with self interaction 1x2x3")
    rs_settings, rs_atoms = rocksalt_with_self_interaction([1, 2, 3], db_name)
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=10)


@pytest.mark.parametrize("basis_func_type", ALL_BASIS_FUNCTIONS)
def test_update_corr_func_spacegroup(db_name, verify_clusters, basis_func_type):
    print("spacegroup")
    sp_settings, sp_atoms = get_spacegroup(db_name, bf_type=basis_func_type)
    do_test_update_correlation_functions(sp_settings, sp_atoms, n_trial_configs=5, fixed=["Ta"])
    verify_clusters(sp_settings)


@pytest.mark.parametrize(
    "settings_maker",
    [
        lambda db_name: rocksalt_with_self_interaction([1, 2, 3], db_name),
        get_spacegroup,
    ],
)
def test_trans_matrix(db_name, settings_maker, verify_clusters):
    settings, atoms = settings_maker(db_name)
    settings.set_active_template(atoms=atoms)
    tm = settings.trans_matrix

    assert len(tm) == len(atoms)
    assert isinstance(tm, TransMatrix)

    for k, v in tm[0].items():
        assert k == v

    verify_clusters(settings)


@pytest.mark.parametrize(
    "filename, settings_maker",
    [
        ("binary_trans_matrix.json", get_binary),
        ("ternary_trans_matrix.json", get_ternary),
        ("spacegroup_trans_matrix.json", get_spacegroup),
    ],
)
def test_saved_spacegroup_trans_matrix(
    db_name, filename, settings_maker, save_trans_matrix, load_trans_matrix
):
    """Test that we receive the same trans matrix mapping as against
    a previously known state."""
    settings, atoms = settings_maker(db_name)
    tm = settings.trans_matrix
    if UPDATE_TRANS_MATRIX:
        # Update the compare trans matrix?
        save_trans_matrix(filename, tm)

    tm_loaded = load_trans_matrix(filename)
    assert tm_loaded == pytest.approx(tm)


def test_init_large_cell(db_name):
    print("Init large cell")
    rs_settings, _ = rocksalt_with_self_interaction([1, 2, 3], db_name)

    atoms = bulk("LiO", crystalstructure="rocksalt", a=4.05, cubic=True)
    atoms = atoms * (2, 2, 2)
    eci = generate_ex_eci(rs_settings)

    # Use quick way of initialisation object
    atoms = attach_calculator(settings=rs_settings, atoms=atoms, eci=eci)

    cf = CorrFunction(rs_settings)
    init_cf = atoms.calc.init_cf

    final_cf = cf.get_cf(atoms)
    for k, v in final_cf.items():
        assert v == pytest.approx(init_cf[k])

    # Try some swaps
    num_X = 0
    num_Mn = 0
    for atom in atoms:
        if atom.symbol == "Li" and num_X < 3:
            atom.symbol = "X"
            num_X += 1
        elif atom.symbol == "Li" and num_Mn < 4:
            atom.symbol = "Mn"
            num_Mn += 1
    atoms.get_potential_energy()

    final_cf = cf.get_cf(atoms)
    calc_cf = atoms.calc.get_cf()
    for k, v in final_cf.items():
        assert v == pytest.approx(calc_cf[k])


def test_4body_attach(db_name):
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    settings = CEBulk(
        crystalstructure="fcc",
        a=4.0,
        size=[2, 2, 2],
        concentration=conc,
        db_name=db_name,
        max_cluster_dia=[6.0, 5.0, 5.0],
    )

    cf = CorrFunction(settings).get_cf(settings.atoms)
    eci = {k: 0.0 for k in cf.keys()}
    eci["c0"] = 1.0
    atoms = settings.atoms.copy() * (3, 3, 3)

    # Simply confirm that no exception is raised.
    # In the past, this failed.
    _ = attach_calculator(settings, atoms=atoms, eci=eci)


def test_with_system_changes_context(db_name):
    settings, atoms = get_binary(db_name)

    atoms.symbols = "Au"

    calc = Clease(settings, eci=generate_ex_eci(settings))
    atoms.calc = calc

    cf = CorrFunction(settings)

    init_cf = cf.get_cf(atoms)

    # Insert 2 Cu atoms
    system_changes = [
        SystemChange(index=0, old_symb="Au", new_symb="Cu", name=""),
        SystemChange(index=1, old_symb="Au", new_symb="Cu", name=""),
    ]
    assert atoms.calc.energy is None
    with atoms.calc.with_system_changes(system_changes) as keeper:
        keeper.keep_changes = False  # Flag to revert

        # We should have 2 Cu atoms now
        num_cu = sum(1 for atom in atoms if atom.symbol == "Cu")
        assert num_cu == 2
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert init_cf != atoms.calc.get_cf()
    # Check that everything is restored to the previous state
    assert all(atoms.symbols == "Au")  # Only Au atoms now
    restored_cf = atoms.calc.get_cf()

    for k, v in init_cf.items():
        assert v == pytest.approx(restored_cf[k])

    # Test that we correctly restore the energy after runs
    energy = atoms.get_potential_energy()
    assert atoms.calc.energy is not None
    with atoms.calc.with_system_changes(system_changes) as keeper:
        keeper.keep_changes = False  # Flag to revert
        assert energy != atoms.get_potential_energy()
    assert energy == pytest.approx(atoms.get_potential_energy())

    # Test we get the same energy after a rerun (same CF's)
    assert not atoms.calc.calculation_required(atoms, ["energy"])
    calc.reset()
    assert atoms.calc.calculation_required(atoms, ["energy"])
    assert energy == pytest.approx(atoms.get_potential_energy())

    # Test that we can keep changes
    with atoms.calc.with_system_changes(system_changes) as keeper:
        # Default is to keep changes. Check that we don't need to do a calculation
        # since the energy is update when we enter the context
        assert not atoms.calc.calculation_required(atoms, ["energy"])
        energy = atoms.get_potential_energy()
    calc.reset()
    assert energy == pytest.approx(atoms.get_potential_energy())


def test_get_ce_energy(db_name):
    settings, atoms = get_binary(db_name)
    assert atoms is not None
    assert settings is not None

    # simple test to receive float energy value
    energy = get_ce_energy(settings, atoms, eci=generate_ex_eci(settings))

    assert isinstance(energy, float)


def test_formula_after_attach(db_name):
    settings, atoms = get_binary(db_name)
    atoms = atoms * (2, 1, 1)
    eci = {
        "c0": 0.0,
        "c1_0": 0.1,
        "c2_d0000_0_00": 0.0,
    }
    atoms_with_calc = attach_calculator(settings, atoms, eci)

    # First, get energy
    assert atoms_with_calc.get_chemical_formula() == "Au28Cu26"

    # Consistency check
    E1 = atoms_with_calc.get_potential_energy()

    # Make sure that no calc was attached to the original object
    assert atoms.calc is None

    # Insert 1 Cu atom
    assert atoms[0].symbol == "Au"
    atoms[0].symbol = "Cu"

    atoms_with_calc = attach_calculator(settings, atoms, eci)
    assert atoms_with_calc.get_chemical_formula() == "Au27Cu27"

    assert E1 != pytest.approx(atoms_with_calc.get_potential_energy(), abs=0.01)

    # Revert when the calculator is attached
    atoms_with_calc[0].symbol = "Au"
    assert E1 == pytest.approx(atoms_with_calc.get_potential_energy(), abs=1e-6)


@pytest.mark.parametrize("rep", [(1, 1, 1), (3, 3, 4), (5, 1, 1)])
def test_non_bkg_indices(db_name, rep):
    settings, atoms = get_rocksalt(db_name)
    atoms = atoms * rep
    atoms = wrap_and_sort_by_position(atoms)
    settings.set_active_template(atoms=atoms)

    bkg_exp = [at.index for at in atoms if at.symbol == "O"]
    assert settings.background_indices == bkg_exp
    non_bkg_exp = [at.index for at in atoms if at.symbol != "O"]
    assert settings.non_background_indices == non_bkg_exp
