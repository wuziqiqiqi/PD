from clease.cluster_generator import ClusterGenerator
import unittest
from ase.build import bulk
import numpy as np


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


if __name__ == '__main__':
    unittest.main()
