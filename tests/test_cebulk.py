"""Test to initiatialize CE using a CEBulk.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""

import os
import json
from clease import CEBulk, CorrFunction, NewStructures, Evaluate, Concentration
from clease.newStruct import MaxAttemptReachedError
from clease.tools import update_db
from ase.calculators.emt import EMT
from ase.db import connect
from ase.build import bulk
from reference_corr_funcs_bulk import all_cf
from ase.build import make_supercell
import numpy as np
import unittest
from unittest.mock import patch

# If this is True, the JSON file containing the correlation functions
# Used to check consistency of the reference functions is updated
# This should normally be False
update_reference_file = False
tol = 1E-9

def get_figures_of_family(setting, cname):
    """Return the figures of a given cluster family."""
    figures = []
    clusters = setting.cluster_list.get_by_name(cname)
    for cluster in clusters:
        figures.append(cluster.indices)
    return figures

def calculate_cf(setting, atoms):
    cf = CorrFunction(setting)
    cf_dict = cf.get_cf(atoms)
    return cf_dict


class TestCEBulk(unittest.TestCase):
    def test_load_from_db(self):
        db_name = 'test_load_from_db.db'
        basis_elements = [['Au', 'Cu']]
        concentration = Concentration(basis_elements=basis_elements)
        setting = CEBulk(crystalstructure='fcc', a=4.05, size=[1, 1, 1],
                         concentration=concentration, db_name=db_name,
                         max_cluster_dia=[4.3, 4.3, 4.3],
                         max_cluster_size=4)
        orig_atoms = setting.atoms.copy()
        atoms = bulk('Au', crystalstructure='fcc', a=4.05, cubic=True)
        setting.set_active_template(atoms=atoms, generate_template=True)

        # Try to read back the old atoms
        setting.set_active_template(atoms=orig_atoms)
        os.remove(db_name)

    def test_corrfunc(self):
        db_name = "test_bulk_corrfunc.db"
        basis_elements = [['Au', 'Cu']]
        concentration = Concentration(basis_elements=basis_elements)
        setting = CEBulk(crystalstructure='fcc', a=4.05, size=[3, 3, 3],
                         concentration=concentration, db_name=db_name,
                         max_cluster_dia=[4.3, 4.3, 4.3],
                         max_cluster_size=4)
        atoms = setting.atoms.copy()
        atoms[0].symbol = 'Cu'
        atoms[3].symbol = 'Cu'
        cf = calculate_cf(setting, atoms)

        if update_reference_file:
            all_cf["binary_fcc"] = cf
        for key in cf.keys():
            self.assertAlmostEqual(cf[key], all_cf["binary_fcc"][key])

        os.remove(db_name)

        basis_elements = [['Li', 'V'], ['X', 'O']]
        concentration = Concentration(basis_elements=basis_elements)
        setting = CEBulk(crystalstructure="rocksalt",
                         a=4.0,
                         size=[2, 2, 1],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[4.01, 4.01])
        atoms = setting.atoms.copy()
        Li_ind = [atom.index for atom in atoms if atom.symbol == 'Li']
        X_ind = [atom.index for atom in atoms if atom.symbol == 'X']
        atoms[Li_ind[0]].symbol = 'V'
        atoms[X_ind[0]].symbol = 'O'
        cf = calculate_cf(setting, atoms)
        if update_reference_file:
            all_cf["two_basis"] = cf
        for key in cf.keys():
            self.assertAlmostEqual(cf[key], all_cf["two_basis"][key])
        os.remove(db_name)

        basis_elements = [['Na', 'Cl'], ['Na', 'Cl']]
        concentration = Concentration(basis_elements=basis_elements,
                                      grouped_basis=[[0, 1]])
        setting = CEBulk(crystalstructure="rocksalt",
                         a=4.0,
                         size=[2, 2, 1],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[4.01, 4.01])
        atoms = setting.atoms.copy()
        atoms[1].symbol = 'Cl'
        atoms[7].symbol = 'Cl'
        cf = calculate_cf(setting, atoms)
        if update_reference_file:
            all_cf["one_grouped_basis"] = cf
        for key in cf.keys():
            self.assertAlmostEqual(cf[key], all_cf["one_grouped_basis"][key])
        os.remove(db_name)

        basis_elements = [['Ca'], ['O', 'F'], ['O', 'F']]
        concentration = Concentration(basis_elements=basis_elements,
                                      grouped_basis=[[0], [1, 2]])
        setting = CEBulk(crystalstructure="fluorite",
                         a=4.0,
                         size=[2, 2, 2],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[4.01, 4.01],
                         ignore_background_atoms=True)
        atoms = setting.atoms.copy()
        O_ind = [atom.index for atom in atoms if atom.symbol == 'O']
        atoms[O_ind[0]].symbol = 'F'
        atoms[O_ind[1]].symbol = 'F'
        cf = calculate_cf(setting, atoms)
        if update_reference_file:
            all_cf["two_grouped_basis_bckgrnd"] = cf

        for key in cf.keys():
            self.assertAlmostEqual(
                cf[key], all_cf["two_grouped_basis_bckgrnd"][key]
            )
        os.remove(db_name)

    def test_binary_system(self):
        """Verifies that one can run a CE for the binary Au-Cu system.

        The EMT calculator is used for energy calculations
        """
        db_name = "test_bulk_binary_system.db"
        basis_elements = [['Au', 'Cu']]
        concentration = Concentration(basis_elements=basis_elements)
        bc_setting = CEBulk(crystalstructure='fcc', a=4.05, size=[3, 3, 3],
                            concentration=concentration,
                            db_name=db_name)

        newstruct = NewStructures(bc_setting, struct_per_gen=3)
        newstruct.generate_initial_pool()

        # Compute the energy of the structures
        calc = EMT()
        database = connect(db_name)

        # Write the atoms to the database
        # for atoms, kvp in zip(all_atoms, key_value_pairs):
        for row in database.select([("converged", "=", False)]):
            atoms = row.toatoms()
            atoms.set_calculator(calc)
            atoms.get_potential_energy()
            update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
        # Evaluate
        Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)

        # Test subclusters for pairs
        for cluster in bc_setting.cluster_list.get_by_size(2):
            sub_cl = bc_setting.cluster_list.get_subclusters(cluster)
            sub_cl_name = set([c.name for c in sub_cl])
            self.assertTrue(sub_cl_name == set(["c0", "c1"]))


        # Test a few known clusters. Triplet nearest neighbour
        name = "c3_d0000_0"
        triplet = bc_setting.cluster_list.get_by_name(name)[0]
        sub_cl = bc_setting.cluster_list.get_subclusters(triplet)
        sub_cl_name = set([c.name for c in sub_cl])
        self.assertTrue(sub_cl_name == set(["c0", "c1", "c2_d0000_0"]))

        name = "c3_d0001_0"
        triplet = bc_setting.cluster_list.get_by_name(name)[0]
        sub_cl = (bc_setting.cluster_list.get_subclusters(triplet))
        sub_cl_name = set([c.name for c in sub_cl])
        self.assertTrue(sub_cl_name == set(["c0", "c1", "c2_d0000_0", "c2_d0001_0"]))

        name = "c4_d0000_0"
        quad = bc_setting.cluster_list.get_by_name(name)[0]
        sub_cl = bc_setting.cluster_list.get_subclusters(quad)
        sub_cl_name = set([c.name for c in sub_cl])
        self.assertTrue(sub_cl_name == set(["c0", "c1", "c2_d0000_0", "c3_d0000_0"]))

        # Try to insert an atoms object with a strange
        P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        self.assertGreater(np.linalg.det(P), 0)
        atoms = make_supercell(bc_setting.prim_cell, P)

        atoms[0].symbol = 'Cu'
        newstruct.insert_structure(init_struct=atoms, generate_template=True)

        # Test that the generate with many templates works
        newstruct.generate_gs_structure_multiple_templates(
            num_prim_cells=16, num_steps_per_temp=100,
            eci={'c0': 1.0, 'c3_d0000_0_000': -0.1}, num_templates=2)
        os.remove(db_name)

    def test_initial_pool(self):
        db_name = "test_bulk_initial_pool.db"
        basis_elements = [['Li', 'V'], ['X', 'O']]
        concentration = Concentration(basis_elements=basis_elements)

        setting = CEBulk(crystalstructure="rocksalt",
                         a=4.0,
                         size=[2, 2, 1],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[4.0, 4.0])
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_initial_pool()

        # At this point there should be the following
        # structures in the DB
        expected_names = ["V1_O1_0", "Li1_X1_0",
                          "V1_X1_0", "Li1_O1_0"]
        db = connect(db_name)
        for name in expected_names:
            num = sum(1 for row in db.select(name=name))
            self.assertEqual(num, 1)
        os.remove(db_name)

    def test_1grouped_basis_probe(self):
        """Test a case where a grouped_basis is used with supercell."""
        # ------------------------------- #
        # 1 grouped basis                 #
        # ------------------------------- #
        # initial_pool + probe_structures #
        # ------------------------------- #
        db_name = "test_bulk_1grouped_probe.db"
        basis_elements = [['Na', 'Cl'], ['Na', 'Cl']]
        concentration = Concentration(basis_elements=basis_elements,
                                      grouped_basis=[[0, 1]])
        setting = CEBulk(crystalstructure="rocksalt",
                         a=4.0,
                         size=[2, 2, 1],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[4.0, 4.0])

        self.assertEqual(setting.num_basis, 1)
        self.assertEqual(len(setting.index_by_basis), 1)
        self.assertTrue(setting.spin_dict == {'Cl': 1.0, 'Na': -1.0})
        self.assertEqual(len(setting.basis_functions), 1)
        try:
            ns = NewStructures(setting=setting, struct_per_gen=2)
            ns.generate_random_structures()
            ns.generate_initial_pool()
            ns = NewStructures(setting=setting, struct_per_gen=2)
            ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                        num_temp=5, num_steps_per_temp=100,
                                        approx_mean_var=True)

        except MaxAttemptReachedError as exc:
            print(str(exc))

        os.remove(db_name)

    def test_2grouped_basis_probe(self):
        # ------------------------------- #
        # 2 grouped basis                 #
        # ------------------------------- #
        # initial_pool + probe_structures #
        # ------------------------------- #
        db_name = "test_bulk_2grouped_probe.db"
        basis_elements = [['Zr', 'Ce'], ['O'], ['O']]
        concentration = Concentration(basis_elements=basis_elements,
                                      grouped_basis=[[0], [1, 2]])
        setting = CEBulk(crystalstructure="fluorite",
                         a=4.0,
                         size=[2, 2, 3],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=2,
                         max_cluster_dia=[4.01])
        fam_figures = get_figures_of_family(setting, "c2_d0005_0")
        self.assertEqual(len(fam_figures[0]), 6)
        self.assertEqual(len(fam_figures[1]), 6)
        self.assertEqual(len(fam_figures[2]), 6)
        self.assertEqual(setting.num_basis, 2)
        self.assertEqual(len(setting.index_by_basis), 2)
        self.assertTrue(setting.spin_dict == {'Ce': 1.0, 'O': -1.0, 'Zr': 0})
        self.assertEqual(len(setting.basis_functions), 2)

        try:
            ns = NewStructures(setting=setting, struct_per_gen=2)
            ns.generate_initial_pool()
            ns = NewStructures(setting=setting, struct_per_gen=2)
            ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                        num_temp=5, num_steps_per_temp=100,
                                        approx_mean_var=True)

        except MaxAttemptReachedError as exc:
            print(str(exc))

        # Try to create a cell with previously failing size
        size = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        atoms = make_supercell(setting.prim_cell, size)

        # This will fail if coordinatation number is wrong
        setting.set_active_template(atoms=atoms, generate_template=True)
        os.remove(db_name)

    def test_2grouped_basis_bckgrnd_probe(self):
        # ---------------------------------- #
        # 2 grouped_basis + background atoms #
        # ---------------------------------- #
        # initial_pool + probe_structures    #
        # ---------------------------------- #
        db_name = "test_bulk_2grouped_bck_probe.db"
        basis_elements = [['Ca'], ['O', 'F'], ['O', 'F']]
        concentration = Concentration(basis_elements=basis_elements,
                                      grouped_basis=[[0], [1, 2]])
        setting = CEBulk(crystalstructure="fluorite",
                         a=4.0,
                         size=[2, 2, 2],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[4.01, 4.01],
                         ignore_background_atoms=True)
        self.assertEqual(setting.num_basis, 2)
        self.assertEqual(len(setting.index_by_basis), 2)
        self.assertTrue(setting.spin_dict == {'F': 1.0, 'O': -1.0})
        self.assertEqual(len(setting.basis_functions), 1)

        try:
            ns = NewStructures(setting=setting, struct_per_gen=2)
            ns.generate_initial_pool()
            ns = NewStructures(setting=setting, struct_per_gen=2)
            ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                        num_temp=5, num_steps_per_temp=100,
                                        approx_mean_var=True)

        except MaxAttemptReachedError as exc:
            print(str(exc))

        os.remove(db_name)

    @patch('test_cebulk.CEBulk._read_data')
    @patch('test_cebulk.CEBulk._store_data')
    @patch('test_cebulk.CEBulk.create_cluster_list_and_trans_matrix')
    def test_fcc_binary_fixed_conc(self, *args):
        # c_Au = 1/3 and c_Cu = 2/3
        A_eq = [[2, -1]]
        b_eq = [0]
        db_name = 'test_fcc_binary_fixed_conc.db'
        conc = Concentration(basis_elements=[['Au', 'Cu']],
                             A_eq=A_eq, b_eq=b_eq)
        setting = CEBulk(crystalstructure='fcc', a=3.8, supercell_factor=27,
                         max_cluster_dia=5.0, max_cluster_size=3,
                         concentration=conc,
                         db_name=db_name)

        # Loop through templates and check that all satisfy constraints
        for atoms in setting.template_atoms.templates['atoms']:
            num = len(atoms)
            ratio = num/3.0
            self.assertAlmostEqual(ratio, int(ratio))
        os.remove(db_name)

    @patch('test_cebulk.CEBulk._read_data')
    @patch('test_cebulk.CEBulk._store_data')
    @patch('test_cebulk.CEBulk.create_cluster_list_and_trans_matrix')
    def test_rocksalt_conc_fixed_one_basis(self, *args):
        db_name = 'test_rocksalt_fixed_one_basis.db'
        basis_elem = [['Li', 'X'], ['O', 'F']]
        A_eq = [[0, 0, 3, -2]]
        b_eq = [0]
        conc = Concentration(basis_elements=basis_elem,
                             A_eq=A_eq, b_eq=b_eq)
        setting = CEBulk(crystalstructure='rocksalt', a=3.8,
                         supercell_factor=27, max_cluster_dia=5.0,
                         max_cluster_size=3, concentration=conc,
                         db_name=db_name)

        # Loop through and check that num_O sites is divisible by 5
        for atoms in setting.template_atoms.templates['atoms']:
            num_O = sum(1 for atom in atoms if atom.symbol == 'O')
            ratio = num_O/5.0
            self.assertAlmostEqual(ratio, int(ratio))
        os.remove(db_name)

    def tearDown(self):
        if update_reference_file:
            print("Updating the reference correlation function file")
            print("This should normally not be done.")
            with open("reference_corr_funcs_bulk.py", 'w') as outfile:
                json.dump(all_cf, outfile, indent=2, separators=(',', ': '))
        return super().tearDown()


if __name__ == '__main__':
    unittest.main()
