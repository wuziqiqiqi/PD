"""Test case for tha GAFit"""
from clease import GAFit
from clease.ga_fit import SaturatedPopulationError
from clease import CEBulk, NewStructures, Concentration, Evaluate
from clease.tools import update_db
from ase.calculators.emt import EMT
from ase.db import connect
from random import choice
import numpy as np
import os
import unittest


def init_system(db_name):
    basis_elements = [['Au', 'Cu']]
    concentration = Concentration(basis_elements=basis_elements)
    bc_settings = CEBulk(crystalstructure='fcc', a=4.05, size=[3, 3, 3],
                         concentration=concentration, db_name=db_name,
                         max_cluster_size=3, max_cluster_dia=[0, 0, 4.06, 4.06])

    newstruct = NewStructures(bc_settings, struct_per_gen=3)
    newstruct.generate_initial_pool()

    # Insert 30 random configurations
    atoms = bc_settings.atoms.copy()
    symbols = ["Au", "Cu"]
    for _ in range(6):
        for atom in atoms:
            atom.symbol = choice(symbols)
        newstruct.insert_structure(init_struct=atoms)

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
    return bc_settings


class TestGAFit(unittest.TestCase):
    def test_init_from_file(self):
        db_name = 'test_ga_fit.db'
        settings = init_system(db_name)
        backup = "ga_test.csv"
        evaluator = Evaluate(settings)
        selector = GAFit(evaluator.cf_matrix, evaluator.e_dft, fname=backup)
        try:
            selector.run(gen_without_change=3, save_interval=1)
        except SaturatedPopulationError:
            pass
        individuals = selector.individuals

        # Restart from file, this time data will be loaded
        selector = GAFit(evaluator.cf_matrix, evaluator.e_dft, fname=backup)
        self.assertTrue(np.allclose(individuals, selector.individuals))
        os.remove(backup)

    def tearDown(self):
        try:
            os.remove('test_ga_fit.db')
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
