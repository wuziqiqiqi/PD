import unittest
from ase.build import bulk
from clease.cluster import Cluster, ClusterList, ClusterFingerprint, ClusterGenerator
import numpy as np


class TestClusterList(unittest.TestCase):

    def test_equiv_deco(self):
        cluster = Cluster()
        predict_result = [{
            'deco': [1, 2, 3, 4],
            'equiv_site': [[0, 1, 2]],
            'result': [[1, 2, 3, 4], [1, 3, 2, 4], [2, 1, 3, 4], [2, 3, 1, 4], [3, 1, 2, 4],
                       [3, 2, 1, 4]]
        }, {
            'deco': [1, 2, 3, 4],
            'equiv_site': [[0, 3]],
            'result': [[1, 2, 3, 4], [4, 2, 3, 1]],
        }, {
            'deco': [1, 2, 3, 4],
            'equiv_site': [],
            'result': [[1, 2, 3, 4]]
        }]
        method_result = []
        for dict_list in predict_result:
            cluster.equiv_sites = dict_list['equiv_site']
            method_result.append(cluster.equiv_deco(dict_list['deco']))

        for count, result_method in enumerate(method_result):
            self.assertListEqual(predict_result[count]['result'], result_method)

    def test_load(self):
        true_fp = ClusterFingerprint([4.5, 4.3, 2.4, -1.0, -3.4, -1.0])
        true_indices = [[0, 3, 3], [1, 0, 5]]
        true_equiv_sites = [[0, 1]]
        cluster = Cluster(name='c3_d0001_0',
                          size=3,
                          diameter=5.4,
                          fingerprint=true_fp,
                          ref_indx=0,
                          indices=true_indices,
                          equiv_sites=true_equiv_sites,
                          trans_symm_group=0)
        cluster_dict = cluster.todict()
        dict_from_cluster = cluster.load(cluster_dict)
        self.assertIsInstance(dict_from_cluster.fp.fp, list)
        self.assertIsInstance(dict_from_cluster.indices, list)
        self.assertIsInstance(dict_from_cluster.equiv_sites, list)
        self.assertListEqual(dict_from_cluster.fp.fp, true_fp.fp)
        self.assertListEqual(dict_from_cluster.indices, true_indices)
        self.assertListEqual(dict_from_cluster.equiv_sites, true_equiv_sites)


if __name__ == '__main__':
    unittest.main()
