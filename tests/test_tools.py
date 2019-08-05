import unittest
import numpy as np
from ase.build import bulk
from clease.tools import close_to_cubic_supercell


class TestTools(unittest.TestCase):
    def test_cubic_sc_fcc(self):
        a = 4.05
        atoms = bulk("Al", a=a, crystalstructure="fcc")
        sc = close_to_cubic_supercell(atoms)
        expected_cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        self.assertTrue(np.allclose(expected_cell, sc.get_cell()))

if __name__ == '__main__':
    unittest.main()