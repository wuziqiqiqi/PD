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

        # Check equivalent sites
        equiv = extractor.equivalent_sites(clusters[0][0])
        self.assertEqual(equiv[0], [0, 1])

        # Try triplets
        clusters = extractor.extract(ref_indx=ref_indx, size=3, cutoff=5.0)

        # Confirm that all internal distances match
        for cluster in clusters:
            d_ref = sorted(extractor._get_internal_distances(cluster[0]))
            for sub in cluster[1:]:
                d = sorted(extractor._get_internal_distances(sub))
                self.assertTrue(np.allclose(d_ref, d))


if __name__ == '__main__':
    unittest.main()
