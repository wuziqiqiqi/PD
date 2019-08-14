import unittest
from clease.cluster_extractor import ClusterExtractor
from ase.build import bulk
import numpy as np


def internal_dists(sub_cluster, pos):
    """
    Calcualte all internal distances
    """
    dists = []
    for indx in sub_cluster:
        diff = pos[sub_cluster, :] - pos[indx, :]
        dists += np.sqrt(np.sum(diff**2, axis=1)).tolist()
    return sorted(dists)


def internal_cos_angles(sub_cluster, pos):
    """
    Calculate the internal angles
    """
    cos_angles = []
    for indx in sub_cluster:
        diff = pos[sub_cluster, :] - pos[indx, :]
        dot_prod = diff.dot(diff.T)
        cos_angles += dot_prod.ravel().tolist()
    return sorted(cos_angles)


def dist_and_ang_match(cluster, pos):
    """
    Check that all internal distances and all internal angles in the
    subclusters are identical
    """
    ref_dists = internal_dists(cluster[0], pos)
    ref_cos_angles = internal_cos_angles(cluster[0], pos)

    for sub_cl in cluster[1:]:
        dists = internal_dists(sub_cl, pos)
        cos_ang = internal_cos_angles(sub_cl, pos)
        dists_ok = np.allclose(ref_dists, dists)
        cos_ang_ok = np.allclose(ref_cos_angles, cos_ang)

        if not dists_ok or not cos_ang_ok:
            return False
    return True


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

    def test_fluorite(self):
        atoms = bulk('CaFF', crystalstructure='fluorite', a=4.0)
        orig_pos = atoms.get_positions()
        for atom in atoms:
            atom.tag = atom.index
        atoms = atoms*(8, 8, 8)

        # Locate reference indices of each sublattice
        ref_indices = []
        for i in range(3):
            diff = atoms.get_positions() - orig_pos[i, :]
            lengths = np.sum(diff**2, axis=1)
            ref_indices.append(np.argmin(lengths))

        com = np.mean(atoms.get_positions(), axis=0)
        atoms.translate(com)
        atoms.wrap()
        pos = atoms.get_positions()

        extractor = ClusterExtractor(atoms)

        # Try 3-body clusters
        print("Testing 3-body clusters")
        for indx in ref_indices:
            clusters = extractor.extract(ref_indx=indx, size=3, cutoff=5.0)
            for cluster in clusters:
                self.assertTrue(dist_and_ang_match(cluster, pos))

        # Try 4-body clusters
        print("Testing 4-body clusters")
        for indx in ref_indices:
            clusters = extractor.extract(ref_indx=indx, size=4, cutoff=4.0)
            for cluster in clusters:
                self.assertTrue(dist_and_ang_match(cluster, pos))


if __name__ == '__main__':
    unittest.main()
