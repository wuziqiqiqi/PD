import unittest
from ase.db import connect
from ase.calculators.emt import EMT
from ase.build import bulk
from clease import CorrFuncEnergyDataManager
import numpy as np
import os


def create_db(db_name):
    db = connect(db_name)
    atoms = bulk('Au')
    atoms2 = bulk('Au')

    cf_func = {'c0': 0.0, 'c1_1': 1.0, 'c2_d0000_0_00': -1.0}
    for i in range(10):
        db.write(atoms, external_tables={'cf_func': cf_func},
                 final_struct_id=2*(i+1), converged=1)
        atoms2.set_calculator(EMT())
        atoms2.get_potential_energy()
        db.write(atoms2)
    return db


class TestDataManager(unittest.TestCase):
    def test_corr_final_energy(self):
        db_name = 'test_get_data.db'
        create_db(db_name)
        cf_names = ['c0', 'c1_1', 'c2_d0000_0_00']
        manager = CorrFuncEnergyDataManager(db_name, cf_names, 'cf_func')
        X, y = manager.get_data([('converged', '=', 1)])
        os.remove(db_name)

        expect_X = np.zeros((10, 3))
        expect_X[:, 0] = 0.0
        expect_X[:, 1] = 1.0
        expect_X[:, 2] = -1.0
        self.assertTrue(np.allclose(X, expect_X))

        csvfile = 'dataset.csv'
        manager.to_csv(csvfile)

        expect_header = "# c0,c1_1,c2_d0000_0_00,E_DFT (eV/atom)\n"
        with open(csvfile, 'r') as f:
            header = f.readline()

        self.assertEqual(header, expect_header)
        X_read = np.loadtxt(csvfile, delimiter=',')
        os.remove(csvfile)
        self.assertTrue(np.allclose(X, X_read[:, :-1]))
        self.assertTrue(np.allclose(y, X_read[:, -1]))

    def test_get_pattern(self):
        db_name = 'test_get_pattern.db'
        create_db(db_name)
        cf_names = ['c0', 'c1_1', 'c2_d0000_0_00']
        manager = CorrFuncEnergyDataManager(db_name, cf_names, 'cf_func')
        manager.get_data([('converged', '=', 1)])
        os.remove(db_name)

        tests = [
            {
                'pattern': 'c',
                'expect': ['c0', 'c1_1', 'c2_d0000_0_00']
            },
            {
                'pattern': 'c0',
                'expect': ['c0']
            },
            {
                'pattern': 'd00',
                'expect': ['c2_d0000_0_00']
            },
            {
                'pattern': '0',
                'expect': ['c0', 'c2_d0000_0_00']
            }
        ]

        for t in tests:
            res = manager.get_matching_names(t['pattern'])
            self.assertEqual(res, t['expect'])

    def test_get_cols(self):
        db_name = 'test_get_cols.db'
        create_db(db_name)
        cf_names = ['c0', 'c1_1', 'c2_d0000_0_00']
        manager = CorrFuncEnergyDataManager(db_name, cf_names, 'cf_func')
        X, _ = manager.get_data([('converged', '=', 1)])
        os.remove(db_name)

        tests = [
            {
                'names': ['c0', 'c1_1', 'c2_d0000_0_00'],
                'expect': X.copy()
            },
            {
                'names': ['c0'],
                'expect': X[:, 0]
            },
            {
                'names': ['c0', 'c1_1'],
                'expect': X[:, :2]
            },
            {
                'names': ['c0', 'c2_d0000_0_00'],
                'expect': X[:, [0, 2]]
            }
        ]

        for t in tests:
            res = manager.get_cols(t['names'])
            self.assertTrue(np.allclose(res, t['expect']))

if __name__ == '__main__':
    unittest.main()
