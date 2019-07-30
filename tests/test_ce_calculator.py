"""Unit tests for the Clease calculator."""
import os
from random import randint
import numpy as np
from clease import CEBulk, CECrystal, CorrFunction, Concentration
from clease.calculator import Clease
from ase.build import bulk
from ase.spacegroup import crystal
from clease.tools import wrap_and_sort_by_position
import time
import unittest


def generate_ex_eci(setting):
    """Return dummy ECIs. All are set to -0.001."""
    cf = CorrFunction(setting)
    cf = cf.get_cf(setting.atoms)
    eci = {key: -0.001 for key in cf}
    return eci


def get_binary(db_name):
    """Return a simple binary test structure."""
    basis_elements = [["Au", "Cu"]]
    concentration = Concentration(basis_elements=basis_elements)
    bc_setting = CEBulk(crystalstructure="fcc",
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
    return bc_setting, wrap_and_sort_by_position(atoms)


def get_ternary(db_name):
    """Return a ternary test structure."""
    basis_elements = [["Au", "Cu", "Zn"]]
    concentration = Concentration(basis_elements=basis_elements)
    bc_setting = CEBulk(crystalstructure="fcc",
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
    return bc_setting, wrap_and_sort_by_position(atoms)


def get_rocksalt(db_name):
    """Test rocksalt where passed atoms with background_atoms."""
    basis_elements=[['Li', 'X', 'V'], ['O']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure='rocksalt',
                     a=4.05,
                     size=[3, 3, 3],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=[7.0, 7.0],
                     ignore_background_atoms=True)

    atoms = bulk("LiO", crystalstructure="rocksalt", a=4.05)
    atoms = atoms * (3, 3, 3)
    Li_indx = [a.index for a in atoms if a.symbol == 'Li']
    for i in range(18):
        if i < 9:
            atoms[Li_indx[i]].symbol = 'V'
        else:
            atoms[Li_indx[i]].symbol = 'X'
    return setting, wrap_and_sort_by_position(atoms)


def rocksalt_with_self_interaction(size, db_name):
    basis_elements=[['Li', 'Mn', 'X'], ['O', 'X']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure='rocksalt',
                     a=4.05,
                     size=size,
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     basis_function='vandewalle',
                     max_cluster_dia=[7.0, 4.0])
    atoms = setting.atoms.copy()
    return setting, atoms


def get_spacegroup(db_name):
    """Test rocksalt where passed atoms."""
    basis = [(0., 0., 0.),
             (0.3894, 0.1405, 0.),
             (0.201, 0.3461, 0.5),
             (0.2244, 0.3821, 0.)]
    spacegroup = 55
    cellpar = [6.25, 7.4, 3.83, 90, 90, 90]
    size = [2, 2, 2]
    basis_elements = [['O', 'X'], ['O', 'X'], ['O', 'X'], ['Ta']]
    grouped_basis = [[0, 1, 2], [3]]
    concentration = Concentration(basis_elements=basis_elements,
                                  grouped_basis=grouped_basis)

    setting = CECrystal(basis=basis,
                        spacegroup=spacegroup,
                        cellpar=cellpar,
                        size=size,
                        concentration=concentration,
                        db_name=db_name,
                        max_cluster_size=3,
                        max_cluster_dia=[5.0, 5.0])

    atoms = crystal(symbols=['O', 'X', 'O', 'Ta'], basis=basis,
                    spacegroup=spacegroup, cell=None,
                    cellpar=cellpar, ab_normal=(0, 0, 1),
                    size=size, primitive_cell=False)

    return setting, wrap_and_sort_by_position(atoms)


def test_update_correlation_functions(setting, atoms, n_trial_configs=20,
                                      fixed=[]):
    """Perform swaps and check that the correlation functions match.

    The comparison is done by check that each CF in the Clease
    calculator is the same as the ones obtained by direct calculation.
    """
    cf = CorrFunction(setting)

    eci = generate_ex_eci(setting)
    calc = Clease(setting, cluster_name_eci=eci)
    atoms.set_calculator(calc)

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
        brute_force_cf = cf.get_cf_by_cluster_names(atoms, calc.cluster_names,
                                                    return_type="array")
        assert np.allclose(brute_force_cf, calc.cf)
    print(np.mean(timings))


def test_insert_element(setting, atoms, n_trial_configs=20):
    from random import choice
    cf = CorrFunction(setting)
    eci = generate_ex_eci(setting)
    calc = Clease(setting, cluster_name_eci=eci)
    atoms.set_calculator(calc)
    elements = setting.unique_elements
    for _ in range(n_trial_configs):
        indx1 = randint(0, len(atoms)-1)
        symb1 = atoms[indx1].symbol
        symb2 = symb1

        while symb2 == symb1:
            symb2 = choice(elements)
        atoms[indx1].symbol = symb2
        atoms.get_potential_energy()
        brute_force_cf = cf.get_cf_by_cluster_names(atoms, calc.cluster_names)
        calc_cf = calc.get_cf_dict()
        for k in calc_cf.keys():
            if k.startswith("c0") or k.startswith("c1"):
                continue
            assert abs(calc_cf[k] - brute_force_cf[k]) < 1E-6


class TestCECalculator(unittest.TestCase):
    def test_indices_of_changed_symbols(self):
        db_name = 'indices_changes_symbol.db'
        setting, atoms = get_binary(db_name)
        eci = generate_ex_eci(setting)
        calc = Clease(setting, cluster_name_eci=eci)
        atoms.set_calculator(calc)

        changes = [2, 6]
        for ch in changes:
            if atoms[ch].symbol == 'Au':
                atoms[ch].symbol = 'Cu'
            else:
                atoms[ch].symbol = 'Au'
        
        calc_changes = calc.indices_of_changed_atoms
        os.remove(db_name)
        self.assertEqual(calc_changes, changes)

    def test_update_corr_func_binary(self):
        db_name = 'cecalc_corr_func_binary.db'
        print('binary')
        bin_setting, bin_atoms = get_binary(db_name)
        test_update_correlation_functions(bin_setting, bin_atoms,
                                          n_trial_configs=5)
        os.remove(db_name)

    def test_update_corr_func_ternary(self):
        db_name = 'cecalc_corr_func_ternary.db'
        print('ternary')
        tern_setting, tern_atoms = get_ternary(db_name)
        test_update_correlation_functions(tern_setting, tern_atoms, n_trial_configs=5)
        os.remove(db_name)

    def test_update_corr_func_rocksalt(self):
        db_name = 'cecalc_corr_func_rocksalt.db'
        print('rocksalt')
        rs_setting, rs_atoms = get_rocksalt(db_name)
        test_update_correlation_functions(
            rs_setting, rs_atoms, n_trial_configs=5, fixed=['O'])
        os.remove(db_name)

    def test_insert_element_rocksalt_1x1x1(self):
        print('rocksalt with self interaction 1x1x1')
        db_name = 'cecalc_rs_1x1x1.db'
        rs_setting, rs_atoms = rocksalt_with_self_interaction([1, 1, 1], db_name)
        test_insert_element(rs_setting, rs_atoms, n_trial_configs=5)
        os.remove(db_name)

    def test_insert_element_rocksalt_1x1x2(self):
        db_name = 'cecalc_rs_1x1x2.db'
        print('rocksalt with self interaction 1x1x2')
        rs_setting, rs_atoms = rocksalt_with_self_interaction([1, 1, 2], db_name)
        test_insert_element(rs_setting, rs_atoms, n_trial_configs=1)
        os.remove(db_name)

    def test_insert_element_rocksalt_1x1x3(self):
        db_name = 'cecalc_rs_1x1x3.db'
        print('rocksalt with self interaction 1x1x3')
        rs_setting, rs_atoms = rocksalt_with_self_interaction([1, 1, 3], db_name)
        test_insert_element(rs_setting, rs_atoms, n_trial_configs=10)
        os.remove(db_name)

    def test_insert_element_rocksalt_1x2x3(self):
        print('rocksalt with self interaction 1x2x3')
        db_name = 'cecalc_rs_1x2x3.db'
        rs_setting, rs_atoms = rocksalt_with_self_interaction([1, 2, 3], db_name)
        test_insert_element(rs_setting, rs_atoms, n_trial_configs=10)
        os.remove(db_name)

    def test_update_corr_func_spacegroup(self):
        print('spacegroup')
        db_name = 'cecalc_corrfunc_spacegroup.db'
        sp_setting, sp_atoms = get_spacegroup(db_name)
        test_update_correlation_functions(sp_setting, sp_atoms, n_trial_configs=5,
                                          fixed=['Ta'])
        os.remove(db_name)

if __name__ == '__main__':
    unittest.main()
