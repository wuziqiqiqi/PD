import unittest
from ase.build import bulk
from clease.tools import min_distance_from_facet, factorize
from clease.tools import all_integer_transform_matrices


class TestTools(unittest.TestCase):
    def test_min_distance_from_facet(self):
        a = 4.0
        atoms = bulk('Al', crystalstructure='sc', a=a)

        x = [3.0, 3.5, 3.9]
        dist = min_distance_from_facet(x, atoms.get_cell())
        self.assertAlmostEqual(dist, 0.1)

    def test_factorize(self):
        fact = sorted(list(factorize(10)))
        self.assertEqual(fact, [2, 5])

        fact = sorted(list(factorize(16)))
        self.assertEqual(fact, [2, 2, 2, 2])

        fact = sorted(list(factorize(24)))
        self.assertEqual(fact, [2, 2, 2, 3])

    def test_all_int_matrices(self):
        _ = all_integer_transform_matrices(10)


if __name__ == '__main__':
    unittest.main()
