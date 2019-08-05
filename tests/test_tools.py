import unittest
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from clease.tools import close_to_cubic_supercell


class TestTools(unittest.TestCase):
    def test_cubic_sc_fcc(self):
        a = 4.05
        atoms = bulk("Al", a=a, crystalstructure="fcc")
        sc = close_to_cubic_supercell(atoms)
        expected_cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        self.assertTrue(np.allclose(expected_cell, sc.get_cell()))

    def test_sp217(self):
        a = 10.553
        b = 10.553
        c = 10.553
        alpha = 90
        beta = 90
        gamma = 90
        cellpar = [a, b, c, alpha, beta, gamma]
        basis = [(0, 0, 0), (0.324, 0.324, 0.324),
                 (0.3582, 0.3582, 0.0393), (0.0954, 0.0954, 0.2725)]

        atoms = crystal(symbols=['Mg', 'Mg', 'Mg', 'Al'],
                        cellpar=cellpar, spacegroup=217,
                        basis=basis, primitive_cell=True)

        sc = close_to_cubic_supercell(atoms)
        expected_cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        self.assertTrue(np.allclose(expected_cell, sc.get_cell()))

    def test_bcc(self):
        a = 3.8
        atoms = bulk("Fe", a=a, crystalstructure="bcc")
        sc = close_to_cubic_supercell(atoms)
        expected_cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        self.assertTrue(np.allclose(sc.get_cell(), expected_cell))

if __name__ == '__main__':
    unittest.main()