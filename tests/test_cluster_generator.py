from clease.cluster_generator import ClusterGenerator
from clease import CEBulk, Concentration
from clease.tools import wrap_and_sort_by_position
import unittest
from ase.build import bulk
import numpy as np
import os


class TestClusterGenerator(unittest.TestCase):
    def test_sites_cutoff_fcc(self):
        atoms = bulk('Al', a=4.05)
        generator = ClusterGenerator(atoms)
        indices = generator.sites_within_cutoff(3.0, ref_lattice=0)
        indices = list(indices)

        # For FCC there should be 12 sites within the cutoff
        self.assertEqual(len(list(indices)), 12)

        # FCC within 4.1
        indices = list(generator.sites_within_cutoff(4.1, ref_lattice=0))
        self.assertEqual(len(indices), 18)

        # FCC within 5.0
        indices = list(generator.sites_within_cutoff(5.0, ref_lattice=0))
        self.assertEqual(len(indices), 42)

    def test_sites_cutoff_bcc(self):
        a = 3.8
        atoms = bulk('Fe', a=a)
        generator = ClusterGenerator(atoms)

        # Neighbour distances
        nn = np.sqrt(3)*a/2.0
        snn = a
        indices = list(generator.sites_within_cutoff(nn+0.01, ref_lattice=0))
        self.assertEqual(len(indices), 8)
        indices = list(generator.sites_within_cutoff(snn+0.01, ref_lattice=0))
        self.assertEqual(len(indices), 14)

    def test_generate_pairs_fcc(self):
        atoms = bulk('Al', a=4.05)
        generator = ClusterGenerator(atoms)
        clusters, fps = generator.generate(2, 5.0, ref_lattice=0)
        self.assertEqual(len(clusters), 3)
        self.assertEqual(len(fps), 3)

    def test_equivalent_sites(self):
        atoms = bulk('Au', a=3.8)
        generator = ClusterGenerator(atoms)

        # Test pairs
        clusters, fps = generator.generate(2, 6.0)
        for c in clusters:
            equiv = generator.equivalent_sites(c[0])
            self.assertEqual(equiv, [[0, 1]])

        # Test a triplet
        clusters, fps = generator.generate(3, 3.0)

        # For the smalles triplet all sites should be equivalent
        equiv = generator.equivalent_sites(clusters[0][0])
        self.assertEqual(equiv, [[0, 1, 2]])

    def test_get_lattice(self):
        tests = [
            {
                'prim': bulk('Al'),
                'atoms': bulk('Al')*(2, 2, 2),
                'site': 4,
                'lattice': 0
            },
            {
                'prim': bulk('LiX', 'rocksalt', 4.0),
                'atoms': bulk('LiX', 'rocksalt', 4.0)*(1, 2, 3),
                'site': 4,
                'lattice': 0,
            },
            {
                'prim': bulk('LiX', 'rocksalt', 4.0),
                'atoms': bulk('LiX', 'rocksalt', 4.0)*(1, 2, 3),
                'site': 5,
                'lattice': 1,
            }
        ]

        for i, test in enumerate(tests):
            test['atoms'].wrap()
            test['prim'].wrap()
            for at in test['prim']:
                at.tag = at.index
            pos = test['atoms'][test['site']].position
            gen = ClusterGenerator(test['prim'])
            lattice = gen.get_lattice(pos)
            msg = 'Test #{} falied. Expected: {} Got {}'.format(
                i, test['lattice'], lattice)
            self.assertEqual(lattice, test['lattice'], msg=msg)


if __name__ == '__main__':
    unittest.main()
