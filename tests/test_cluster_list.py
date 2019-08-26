import unittest
from ase.build import bulk
from clease.cluster import Cluster
from clease.cluster_list import ClusterList
import numpy as np


class TestClusterList(unittest.TestCase):
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
