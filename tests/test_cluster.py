import unittest
from ase.build import bulk
from clease.cluster import Cluster
from clease.cluster_list import ClusterList
from clease.cluster_fingerprint import ClusterFingerprint
from clease.cluster_generator import ClusterGenerator
import numpy as np


class TestClusterList(unittest.TestCase):
    def test_to_from_dict(self):
        cluster = Cluster()
        predict_result = [{'deco': [1, 2, 3, 4],
                           'equiv_site': [[0, 1, 2]],
                           'result': [[1, 2, 3, 4],
                                      [1, 3, 2, 4],
                                      [2, 1, 3, 4],
                                      [2, 3, 1, 4],
                                      [3, 1, 2, 4],
                                      [3, 2, 1, 4]]},
                          {'deco': [1, 2, 3, 4],
                           'equiv_site': [[0, 3]],
                           'result': [[1, 2, 3, 4],
                                      [4, 2, 3, 1]], },
                          {'deco': [1, 2, 3, 4],
                           'equiv_site': [],
                           'result': [[1, 2, 3, 4]]}
                          ]
        method_result = []
        for dict_list in predict_result:
            cluster.equiv_sites = dict_list['equiv_site']
            method_result.append(cluster.equiv_deco(dict_list['deco']))

        for count, result_method in enumerate(method_result):
            self.assertListEqual(predict_result[count]['result'], result_method)


if __name__ == '__main__':
    unittest.main()
