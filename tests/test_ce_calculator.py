"""Unit tests for the Clease calculator."""
import os
import time
import unittest
import pytest
import numpy as np
from numpy.random import randint, choice
from ase.build import bulk
from ase.spacegroup import crystal

from clease.tools import wrap_and_sort_by_position
from clease.settings import CEBulk, CECrystal
from clease.corr_func import CorrFunction
from clease.settings import Concentration
from clease.calculator import Clease, attach_calculator, get_ce_energy


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
    bc_settings = CEBulk(crystalstructure="fcc",
                         a=4.05,
                         size=[3, 3, 3],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[5.0, 5.0])

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms * (3, 3, 3)
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"
    return bc_settings, wrap_and_sort_by_position(atoms)


def get_ternary(db_name):
    """Return a ternary test structure."""
    basis_elements = [["Au", "Cu", "Zn"]]
    concentration = Concentration(basis_elements=basis_elements)
    bc_settings = CEBulk(crystalstructure="fcc",
                         a=4.05,
                         size=[3, 3, 3],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_dia=[5.0, 5.0],
                         max_cluster_size=3)

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms * (3, 3, 3)
    for i in range(2):
        atoms[3 * i].symbol = "Au"
        atoms[3 * i + 1].symbol = "Cu"
        atoms[3 * i + 2].symbol = "Zn"
    return bc_settings, wrap_and_sort_by_position(atoms)


def get_rocksalt(db_name):
    """Test rocksalt where passed atoms with background_atoms."""
    basis_elements = [['Li', 'X', 'V'], ['O']]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure='rocksalt',
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[7.0, 7.0])

    atoms = bulk("LiO", crystalstructure="rocksalt", a=4.05)
    atoms = atoms * (3, 3, 3)
    Li_indx = [a.index for a in atoms if a.symbol == 'Li']
    for i in range(18):
        if i < 9:
            atoms[Li_indx[i]].symbol = 'V'
        else:
            atoms[Li_indx[i]].symbol = 'X'
    return settings, wrap_and_sort_by_position(atoms)


def rocksalt_with_self_interaction(size, db_name):
    basis_elements = [['Li', 'Mn', 'X'], ['O', 'X']]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure='rocksalt',
                      a=4.05,
                      size=size,
                      concentration=concentration,
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[7.0, 4.0])
    settings.basis_func_type = 'trigonometric'
    atoms = settings.atoms.copy()
    return settings, atoms


def get_spacegroup(db_name):
    """Test rocksalt where passed atoms."""
    basis = [(0., 0., 0.), (0.3894, 0.1405, 0.), (0.201, 0.3461, 0.5), (0.2244, 0.3821, 0.)]
    spacegroup = 55
    cellpar = [6.25, 7.4, 3.83, 90, 90, 90]
    size = [2, 2, 2]
    basis_elements = [['O', 'X'], ['O', 'X'], ['O', 'X'], ['Ta']]
    grouped_basis = [[0, 1, 2], [3]]
    concentration = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    settings = CECrystal(basis=basis,
                         spacegroup=spacegroup,
                         cellpar=cellpar,
                         size=size,
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[5.0, 5.0])
    settings.include_background_atoms = True

    atoms = crystal(symbols=['O', 'X', 'O', 'Ta'],
                    basis=basis,
                    spacegroup=spacegroup,
                    cell=None,
                    cellpar=cellpar,
                    ab_normal=(0, 0, 1),
                    size=size)

    return settings, wrap_and_sort_by_position(atoms)


def do_test_update_correlation_functions(settings, atoms, n_trial_configs=20, fixed=()):
    """Perform swaps and check that the correlation functions match.

    The comparison is done by check that each CF in the Clease
    calculator is the same as the ones obtained by direct calculation.
    """
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
        start = time.time()
        atoms.get_potential_energy()
        timings.append(time.time() - start)
        brute_force_cf = cf.get_cf_by_names(atoms, calc.cf_names)
        calc_cf = calc.get_cf()

        for key in calc_cf.keys():
            assert abs(calc_cf[key] - brute_force_cf[key]) < 1E-6

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
            assert abs(calc_cf[k] - brute_force_cf[k]) < 1E-6


def test_normfactors_no_self_interaction(db_name):
    settings, atoms = get_binary(db_name)

    eci = generate_ex_eci(settings)
    calc = Clease(settings, eci=eci)
    atoms.calc = calc

    for cluster in settings.cluster_list:
        if cluster.name == 'c0' or cluster.name == 'c1':
            continue
        norm_factors = cluster.info['normalization_factor']
        assert np.allclose(norm_factors, 1.0)


def test_indices_of_changed_symbols(db_name):
    settings, atoms = get_binary(db_name)
    eci = generate_ex_eci(settings)
    calc = Clease(settings, eci=eci)
    atoms.calc = calc

    changes = [2, 6]
    for ch in changes:
        if atoms[ch].symbol == 'Au':
            atoms[ch].symbol = 'Cu'
        else:
            atoms[ch].symbol = 'Au'

    calc_changes = calc.indices_of_changed_atoms
    assert calc_changes == changes


def test_update_corr_func_binary(db_name):
    print('binary')
    bin_settings, bin_atoms = get_binary(db_name)
    do_test_update_correlation_functions(bin_settings, bin_atoms, n_trial_configs=5)


def test_update_corr_func_ternary(db_name):
    print('ternary')
    tern_settings, tern_atoms = get_ternary(db_name)
    do_test_update_correlation_functions(tern_settings, tern_atoms, n_trial_configs=5)
    os.remove(db_name)


def test_update_corr_func_rocksalt(db_name):
    print('rocksalt')
    rs_settings, rs_atoms = get_rocksalt(db_name)
    do_test_update_correlation_functions(rs_settings, rs_atoms, n_trial_configs=5, fixed=['O'])


def test_insert_element_rocksalt_1x1x1(db_name):
    print('rocksalt with self interaction 1x1x1')
    rs_settings, rs_atoms = rocksalt_with_self_interaction([1, 1, 1], db_name)
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=5)


def test_insert_element_rocksalt_1x1x2(db_name):
    print('rocksalt with self interaction 1x1x2')
    rs_settings, rs_atoms = rocksalt_with_self_interaction([1, 1, 2], db_name)
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=1)


def test_insert_element_rocksalt_1x1x3(db_name):
    print('rocksalt with self interaction 1x1x3')
    rs_settings, rs_atoms = rocksalt_with_self_interaction([1, 1, 3], db_name)
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=10)


def test_insert_element_rocksalt_1x2x3(db_name):
    print('rocksalt with self interaction 1x2x3')
    rs_settings, rs_atoms = rocksalt_with_self_interaction([1, 2, 3], db_name)
    do_test_insert_element(rs_settings, rs_atoms, n_trial_configs=10)


def test_update_corr_func_spacegroup(db_name):
    print('spacegroup')
    sp_settings, sp_atoms = get_spacegroup(db_name)
    do_test_update_correlation_functions(sp_settings, sp_atoms, n_trial_configs=5, fixed=['Ta'])


def test_init_large_cell(db_name):
    print('Init large cell')
    rs_settings, _ = rocksalt_with_self_interaction([1, 2, 3], db_name)

    atoms = bulk('LiO', crystalstructure='rocksalt', a=4.05, cubic=True)
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
        if atom.symbol == 'Li' and num_X < 3:
            atom.symbol = 'X'
            num_X += 1
        elif atom.symbol == 'Li' and num_Mn < 4:
            atom.symbol = 'Mn'
            num_Mn += 1
    atoms.get_potential_energy()

    final_cf = cf.get_cf(atoms)
    calc_cf = atoms.calc.get_cf()
    for k, v in final_cf.items():
        assert v == pytest.approx(calc_cf[k])


def test_4body_attach(db_name):
    conc = Concentration(basis_elements=[['Au', 'Cu']])
    settings = CEBulk(crystalstructure='fcc',
                      a=4.0,
                      size=[2, 2, 2],
                      concentration=conc,
                      db_name=db_name,
                      max_cluster_size=4,
                      max_cluster_dia=[6.0, 5.0, 5.0])

    cf = CorrFunction(settings).get_cf(settings.atoms)
    eci = {k: 0.0 for k in cf.keys()}
    eci['c0'] = 1.0
    atoms = settings.atoms.copy() * (3, 3, 3)

    # Simply confirm that no exception is raised.
    # In the past, this failed.
    _ = attach_calculator(settings, atoms=atoms, eci=eci)


def test_given_change_and_restore(db_name):
    settings, atoms = get_binary(db_name)

    for atom in atoms:
        atom.symbol = 'Au'

    calc = Clease(settings, eci=generate_ex_eci(settings))
    atoms.calc = calc

    cf = CorrFunction(settings)

    init_cf = cf.get_cf(atoms)

    # Insert to Cu atoms
    _ = atoms.calc.get_energy_given_change([(0, 'Au', 'Cu'), (1, 'Au', 'Cu')])

    # We should have to Cu atoms now
    num_cu = sum(1 for atom in atoms if atom.symbol == 'Cu')
    assert num_cu == 2

    cf_calc_two_inserts = atoms.calc.get_cf()
    cf_scratch = cf.get_cf(atoms)

    for k, v in cf_calc_two_inserts.items():
        assert v == pytest.approx(cf_scratch[k])

    atoms.calc.restore()

    # Now we should be back to pure Au
    num_au = sum(1 for atom in atoms if atom.symbol == 'Au')
    assert num_au == len(atoms)

    cf_calc = calc.get_cf()
    for k, v in cf_calc.items():
        assert v == pytest.approx(init_cf[k])

    # Insert two atoms again
    _ = atoms.calc.get_energy_given_change([(0, 'Au', 'Cu'), (1, 'Au', 'Cu')])

    # Clear the history
    atoms.calc.clear_history()

    # Restore should now not have any effect
    atoms.calc.restore()
    cf_calc = calc.get_cf()

    # Should still be two Cu atoms
    num_cu = sum(1 for atom in atoms if atom.symbol == 'Cu')
    assert num_cu == 2

    for k, v in cf_calc_two_inserts.items():
        assert v == pytest.approx(cf_calc[k])


def test_get_ce_energy(db_name):
    settings, atoms = get_binary(db_name)
    assert atoms is not None
    assert settings is not None

    # simple test to receive float energy value
    energy = get_ce_energy(settings, atoms, eci=generate_ex_eci(settings))
    assert isinstance(energy, float)
