import unittest
from clease.cluster_extractor import ClusterExtractor
from ase.build import bulk
import numpy as np


class TestClusterExtractor(unittest.TestCase):
    def test_fcc(self):
        atoms = bulk("Al", a=4.05)*(10, 10, 10)

        # Find the atoms at the center of the cell
        com = atoms.get_center_of_mass()
        pos = atoms.get_positions() - com
        diff = np.sum(pos**2, axis=1)
        ref_indx = np.argmin(diff)

        extractor = ClusterExtractor(atoms)
        clusters = extractor.extract(ref_indx=ref_indx, size=2, cutoff=5.0)

        self.assertTrue(len(clusters), 3)

        # Check corrcet numbers of first neighbours
        self.assertTrue(len(clusters[0]), 12)

        # Check correct numbers of second neighbours
        self.assertTrue(len(clusters[1]), 6)

        # Check correct numbers os third neighbours
        self.assertTrue(len(clusters[2]), 18)

if __name__ == '__main__':
    unittest.main()
