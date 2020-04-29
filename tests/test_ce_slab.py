import unittest
from ase.build import bulk
import numpy as np
from clease.settings_slab import (
    get_prim_slab_cell, add_vacuum_layers, CESlab
)
from clease import Concentration, settingsFromJSON
import os
import ase


class TestCESlab(unittest.TestCase):
    def test_prim_cell_construction(self):
        tests = [
            {
                'cell': bulk('Al', a=4.05, cubic=True),
                'miller': [1, 1, 1],
                'expect_dist': 4.05/np.sqrt(3.0)
            },
            {
                'cell': bulk('Al', a=4.05, cubic=True),
                'miller': [1, 1, 0],
                'expect_dist': 4.05/np.sqrt(2.0)
            },
            {
                'cell': bulk('Fe', a=4.05, cubic=True),
                'miller': [1, 1, 1],
                'expect_dist': 4.05/np.sqrt(3.0)
            },
            {
                'cell': bulk('Fe', a=4.05, cubic=True),
                'miller': [1, 1, 0],
                'expect_dist': 4.05/np.sqrt(2.0)
            }
        ]

        for i, test in enumerate(tests):
            prim = get_prim_slab_cell(test['cell'], test['miller'])
            dist = prim.get_cell()[2, 2]
            self.assertAlmostEqual(dist, test['expect_dist'])

    def test_add_vacuum_layers(self):
        atoms = bulk('Al', a=4.05, cubic=True)
        prim = get_prim_slab_cell(atoms, [1, 1, 1])
        z_prim = prim.cell[2, 2]
        atoms = prim*(1, 1, 3)
        z_orig = atoms.cell[2, 2]
        atoms = add_vacuum_layers(atoms, prim, 10.0)
        new_z = atoms.cell[2, 2]
        z_vac = int(-(-10.0//z_prim)) * z_prim
        self.assertGreater(z_vac, 10)
        self.assertAlmostEqual(new_z, z_orig+z_vac)

        tol = 1e-6
        for atom in atoms:
            if atom.position[2] > z_orig-tol:
                self.assertEqual(atom.symbol, 'X')
            else:
                self.assertEqual(atom.symbol, 'Al')

    def test_load(self):
        if ase.__version__ < '3.19':
            self.skipTest("CESlab requires ASE > 3.19")
        db_name = 'test_load_save_ceslab.db'
        atoms = bulk('Al', a=4.05, cubic=True)
        conc = Concentration(basis_elements=[['Al', 'X']])
        settings = CESlab(atoms, (1, 1, 1), conc, db_name=db_name)

        backup_file = 'test_save_ceslab.json'
        settings.save(backup_file)

        settings2 = settingsFromJSON(backup_file)

        self.assertEqual(settings.atoms, settings2.atoms)
        self.assertEqual(settings.size, settings2.size)
        self.assertEqual(settings.concentration, settings2.concentration)
        os.remove(db_name)
        os.remove(backup_file)


if __name__ == '__main__':
    unittest.main()
