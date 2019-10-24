import unittest
from ase.build import bulk
from clease.tools import min_distance_from_facet, factorize
from clease.tools import all_integer_transform_matrices
from clease.tools import species_chempot2eci


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

    def test_species_chempot2eci(self):
        tests = [
            {
                'species': {'Au': 1.5},
                'bf_list': [{'Au': 1.0, 'Cu': -1.0}],
                'expect': {'c1_0': 1.5}
            },
            {
                'species': {'Au': 1.5, 'Cu': 0.5},
                'bf_list': [{'Au': 0.3, 'Cu': 1.2, 'X': 3.0},
                            {'Au': -0.3, 'Cu': 1.2, 'X': -3.0}],
                'expect': {'c1_0': 65/24, 'c1_1': -55/24}
            }
        ]

        for i, test in enumerate(tests):
            eci = species_chempot2eci(test['bf_list'], test['species'])
            msg = 'Test #{} failed '.format(i)
            msg += 'Setup: {}'.format(test)
            for k, v in eci.items():
                self.assertAlmostEqual(v, test['expect'][k], msg=msg)

if __name__ == '__main__':
    unittest.main()
