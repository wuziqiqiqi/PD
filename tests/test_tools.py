import unittest
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from clease.tools import min_distance_from_facet


class TestTools(unittest.TestCase):
    def test_min_distance_from_facet(self):
        a = 4.0
        atoms = bulk('Al', crystalstructure='sc', a=a)

        x = [3.0, 3.5, 3.9]
        dist = min_distance_from_facet(x, atoms.get_cell())
        self.assertAlmostEqual(dist, 0.1)


if __name__ == '__main__':
    unittest.main()
