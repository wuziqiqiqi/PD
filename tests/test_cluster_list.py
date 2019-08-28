import unittest
from ase.build import bulk
from clease.cluster import Cluster
from clease.cluster_list import ClusterList
from clease.cluster_fingerprint import ClusterFingerprint
import numpy as np


class TestClusterList(unittest.TestCase):
    def test_to_from_dict(self):
        fp1 = ClusterFingerprint([4.5, 4.3, 2.4, -1.0, -3.4, -1.0])
        cluster1 = Cluster('c3_d0001_0', 3, 5.4, fp1,
                           0, [[0, 3, 3], [1, 0, 5]], [[0, 1]], 0)
        cluster1_dict = cluster1.todict()

        # Make another cluster
        fp2 = ClusterFingerprint([2.0, 1.9, 2.1])
        cluster2 = Cluster('c2_d0002_0', 4, 5.1, fp2,
                           1, [[1, 0], [10, 1]], [], 2)

        # Transfer the properties from cluster1
        cluster2.from_dict(cluster1_dict)

        self.assertEqual(cluster1.name, cluster2.name)
        self.assertEqual(cluster1.size, cluster2.size)
        self.assertAlmostEqual(cluster1.diameter, cluster2.diameter)
        self.assertEqual(cluster1.fp, cluster2.fp)
        self.assertEqual(cluster1.ref_indx, cluster2.ref_indx)
        self.assertEqual(cluster1.indices, cluster2.indices)
        self.assertEqual(cluster1.group, cluster2.group)

    def test_parent_tracker(self):
        atoms = bulk("NaCl", crystalstructure='rocksalt', a=4.0)
        atoms.wrap()
        for atom in atoms:
            atom.tag = atom.index
        orig_pos = atoms.get_positions()
        atoms = atoms*(7, 7, 7)

        # Find ref indices
        ref_indices = []
        for i in range(2):
            diff = atoms.get_positions() - orig_pos[i, :]
            lengths = np.sum(diff**2, axis=1)
            ref_indices.append(np.argmin(diff))

        com = np.mean(atoms.get_positions(), axis=0)
        atoms.translate(com)
        atoms.wrap()

        nn_distance = 2.0
        clist = ClusterList()
        pos = atoms.get_positions()
        for i, ref in enumerate(ref_indices):
            diff = pos - pos[ref, :]
            lengths = np.sqrt(np.sum(diff**2, axis=1))
            indices = np.nonzero(lengths < nn_distance + 0.01)[0].tolist()
            indices.remove(ref)

            # Convert to the required format
            indices = [[x] for x in indices]

            dia = 2.0
            order = [[0, 1] for _ in range(len(indices))]
            equiv_sites = [[0, 1]]
            descriptor = '01nn'
            new_clust = Cluster(2, descriptor, indices, order, equiv_sites, i,
                                2.0, ref)

            clist.append(new_clust)

        # Confirm that it the datastructure knows that the nearest neighbour
        # cluster exists in both basis
        equiv = clist.get_equivalent_clusters(clist.clusters[0])
        self.assertEqual(len(equiv), 2)


if __name__ == '__main__':
    unittest.main()
