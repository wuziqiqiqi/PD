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
            equiv_sites = [[0, 1]]
            name = '01nn'
            fp = ClusterFingerprint([4.5, 4.3, 2.4, -1.0, -3.4, -1.0])
            new_clust = Cluster(name, 2, dia, fp, ref, indices, equiv_sites, i)

            clist.append(new_clust)

        # Confirm that it the datastructure knows that the nearest neighbour
        # cluster exists in both basis
        equiv = clist.get_equivalent_clusters(clist.clusters[0])
        self.assertEqual(len(equiv), 2)

    def test_cf_names_no_equiv_sites(self):
        fp = ClusterFingerprint([3.4, 4.3, 1.2, -1.0, 2.0, 3.0])
        prefix = 'c3_d0001_0'
        cluster = Cluster(prefix, 3, 5.4, fp, 0,
                          [[1, 0,], [3, 4]], [], 0)

        cf_names = ClusterList.get_cf_names(cluster, 2)
        expected_suffix = ['_000', '_001', '_010', '_011', '_100', '_101',
                           '_110', '_111']
        expected_cf_names = [prefix + s for s in expected_suffix]
        self.assertEqual(cf_names, expected_cf_names)

    def test_cf_names_with_equiv_sites(self):
        fp = ClusterFingerprint([3.4, 4.3, 1.2, -1.0, 2.0, 3.0])
        prefix = 'c3_d0001_0'
        cluster = Cluster(prefix, 3, 5.4, fp, 0,
                          [[1, 0,], [3, 4]], [[0, 1]], 0)

        cf_names = ClusterList.get_cf_names(cluster, 2)
        expected_suffix = ['_000', '_001', '_010', '_011', '_110', '_111']
        expected_cf_names = [prefix + s for s in expected_suffix]
        self.assertEqual(cf_names, expected_cf_names)

    def test_get_all_cf_names(self):
        cluster_list = ClusterList()
        fp = ClusterFingerprint([3.4, 4.3, 1.2, -1.0, 2.0, 3.0])
        prefix1 = 'c3_d0001_0'
        cluster1 = Cluster(prefix1, 3, 5.4, fp, 0,
                           [[1, 0,], [3, 4]], [[0, 1]], 0)

        fp = ClusterFingerprint([3.4, 4.3, 1.2, -1.0, 2.0, 3.0])
        prefix2 = 'c3_d0002_0'
        cluster2 = Cluster(prefix2, 3, 5.4, fp, 0,
                           [[1, 0,], [3, 4]], [], 0)

        cluster_list.append(cluster1)
        cluster_list.append(cluster2)
        all_cf_names = cluster_list.get_all_cf_names(2)

        expected_suffix1 = ['_000', '_001', '_010', '_011', '_110', '_111']
        expected_cf_names1 = [prefix1 + s for s in expected_suffix1]
        expected_suffix2 = ['_000', '_001', '_010', '_011', '_100', '_101',
                           '_110', '_111']
        expected_cf_names2 = [prefix2 + s for s in expected_suffix2]

        expected_cf_names = expected_cf_names1 + expected_cf_names2
        self.assertEqual(all_cf_names, expected_cf_names)

    def test_unique_indices(self):
        cluster1 = Cluster(indices=[[0, 3, 3], [9, 9, 9]])
        cluster2 = Cluster(indices=[[10, 12], [2, 5]])
        cluster_list = ClusterList()
        cluster_list.append(cluster1)
        cluster_list.append(cluster2)
        indices = cluster_list.unique_indices
        indices.sort()
        expected = [0, 2, 3, 5, 9, 10, 12]
        self.assertEqual(indices, expected)

    def test_unique_indices_per_symm_group(self):
        cluster1 = Cluster(indices=[[0, 3, 3], [9, 9, 9]], trans_symm_group=0)
        cluster2 = Cluster(indices=[[10, 12], [2, 5]], trans_symm_group=0)
        cluster3 = Cluster(indices=[[0, 3, 3], [9, 9, 9]], trans_symm_group=1)
        cluster4 = Cluster(indices=[[10, 12], [2, 5]], trans_symm_group=2)

        cluster_list = ClusterList()
        cluster_list.append(cluster1)
        cluster_list.append(cluster2)
        cluster_list.append(cluster3)
        cluster_list.append(cluster4)

        indices = cluster_list.unique_indices_per_group
        indices = [sorted(x) for x in indices]
        expected = [[0, 2, 3, 5, 9, 10, 12], [0, 3, 9], [2, 5, 10, 12]]
        self.assertEqual(indices, expected)

    def test_subcluster(self):
        triplet = Cluster(size=3, indices=[[0, 3, 5], [1, 4, 6]],
                          trans_symm_group=2)
        doublet = Cluster(size=2, indices= [[0, 3], [10, 12]], 
                       trans_symm_group=2)
        self.assertTrue(doublet.is_subcluster(triplet))
        self.assertFalse(triplet.is_subcluster(doublet))

        doublet = Cluster(size=2, indices=[[0, 8], [10, 12]],
                          trans_symm_group=2)
        self.assertFalse(doublet.is_subcluster(triplet))

    def test_get_subclusters(self):
        trip = Cluster(size=3, indices=[[0, 3, 5], [1, 4, 6]],
                       trans_symm_group=2)
        d1 = Cluster(size=2, indices=[[0, 3], [10, 12]], trans_symm_group=2)
        d2 = Cluster(size=2, indices=[[0, 8], [10, 12]], trans_symm_group=2)
        d3 = Cluster(size=2, indices=[[0, 5], [4, 6]], trans_symm_group=2)
        cluster_list = ClusterList()
        cluster_list.append(trip)
        cluster_list.append(d1)
        cluster_list.append(d2)
        cluster_list.append(d3)

        subclusters = cluster_list.get_subclusters(trip)
        expected = [d1, d3]
        self.assertEqual(expected, subclusters)

    def test_get_key(self):
        cluster = Cluster(size=3, indices=[[3, 0, 5], [1, 4, 6]], equiv_sites=[])
        key = cluster.get_figure_key([3, 0, 5])
        self.assertEqual(key, '3-0-5')

        cluster.equiv_sites = [[0, 1]]
        key = cluster.get_figure_key([3, 0, 5])
        self.assertEqual(key, '0-3-5')

        # Try to pass a figure that mimics translated indices
        key = cluster.get_figure_key([6, 4, 0])
        self.assertEqual(key, '4-6-0')

    def test_num_occurences(self):
        c1 = Cluster(size=3, indices=[[3, 0, 0], [0, 3, 0], [0, 4, 5]],
                     equiv_sites=[])
        occ_count = c1.num_fig_occurences
        expected = {'3-0-0': 1, '0-3-0': 1, '0-4-5': 1}
        for k in expected.keys():
            self.assertEqual(occ_count[k], expected[k])

    def test_num_occurences_equiv_sites(self):
        c1 = Cluster(size=3, indices=[[3, 0, 0], [0, 3, 0], [0, 4, 5]],
                     equiv_sites=[[0, 1]])
        occ_count = c1.num_fig_occurences
        expected = {
            '0-3-0': 2,
            '0-4-5': 1
        }
        for k in expected.keys():
            self.assertEqual(occ_count[k], expected[k])
        

if __name__ == '__main__':
    unittest.main()
