import unittest
from ase.db import connect
from ase.build import bulk
from clease.db_util import get_all_cf_names, get_all_cf, get_cf_tables
import os
import subprocess


class TestDBUtil(unittest.TestCase):
    def test_get_all_cf_names(self):
        db_name = 'test_get_all_cf_names.db'
        cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
        db = connect(db_name)
        for i in range(10):
            db.write(bulk('Cu'), external_tables={'polynomial_cf': cf})

        names = get_all_cf_names(db_name, 'polynomial_cf')
        self.assertEqual(sorted(names), sorted(list(cf.keys())))
        os.remove(db_name)

    def test_get_cf(self):
        db_name = 'test_get_cf.db'
        cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
        db = connect(db_name)
        for i in range(10):
            db.write(bulk('Cu'), external_tables={'polynomial_cf': cf})

        cf_from_db = get_all_cf(db_name, 'polynomial_cf', 2)
        self.assertDictEqual(cf_from_db, cf)
        os.remove(db_name)

    def test_get_cf_tables(self):
        db_name = 'test_get_cf_tables.db'
        cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
        cf2 = {'c0': 0.1, 'c1_1': 0.5, 'c2_d0000_0_00': -0.2}
        db = connect(db_name)
        for i in range(10):
            db.write(bulk('Cu'), external_tables={'polynomial_cf': cf,
                                                  'trigonometric_cf': cf2})

        cf_tab = get_cf_tables(db_name)
        expect_tab = ['polynomial_cf', 'trigonometric_cf']
        self.assertEqual(sorted(cf_tab), expect_tab)
        os.remove(db_name)

    def test_cli_runs(self):
        db_name = 'test_cli_runs.db'
        cf = {'c0': 1.0, 'c1_1': 0.5, 'c2_d0000_0_00': 1.0}
        cf2 = {'c0': 0.1, 'c1_1': 0.5, 'c2_d0000_0_00': -0.2}
        db = connect(db_name)
        for i in range(10):
            db.write(bulk('Cu'), external_tables={'polynomial_cf': cf,
                                                  'trigonometric_cf': cf2})

        cmds = [["clease", "db", "-h"],
                ["clease", "db", db_name, "--show", "tab"],
                ["clease", "db", db_name, "--show", "names"],
                ["clease", "db", db_name, "--show", "cf", "--id", "1"]]
        for cmd in cmds:
            return_code = subprocess.call(cmd)
            self.assertEqual(return_code, 0)
        os.remove(db_name)


if __name__ == '__main__':
    unittest.main()
