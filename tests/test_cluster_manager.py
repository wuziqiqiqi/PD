import unittest
from ase.build import bulk
from clease import ClusterManager
import numpy as np


class TestClusterManager(unittest.TestCase):
    def trans_matrix_matches(self, tm, template):
        for row, atom in zip(tm, template):
            for k, v in row.items():
                d_orig = template.get_distance(
                    0, int(k), mic=True, vector=True)
                d = template.get_distance(atom.index, v, vector=True, mic=True)
                if not np.allclose(d, d_orig):
                    return False
        return True

    def test_tm_fcc(self):
        prim = bulk('Al')
        prim[0].tag = 0
        manager = ClusterManager(prim)
        template = prim*(3, 3, 3)
        template.wrap()
        manager.build(max_size=2, max_cluster_dia=3.0)
        trans_mat = manager.translation_matrix(template)
        self.assertTrue(self.trans_matrix_matches(trans_mat, template))

    def test_tm_hcp(self):
        prim = bulk('Mg', crystalstructure='hcp', a=3.8, c=4.8)
        prim.wrap()
        for atom in prim:
            atom.tag = atom.index
        template = prim*(3, 3, 3)
        template.wrap()
        manager = ClusterManager(prim)
        manager.build(max_size=2, max_cluster_dia=4.0)
        tm = manager.translation_matrix(template)
        self.assertTrue(self.trans_matrix_matches(tm, template))

    def test_tm_rocksalt(self):
        prim = bulk('LiX', crystalstructure='rocksalt', a=4.0)
        prim.wrap()
        for atom in prim:
            atom.tag = atom.index

        template = prim*(2, 2, 2)
        manager = ClusterManager(prim)
        manager.build(max_size=2, max_cluster_dia=3.0)
        tm = manager.translation_matrix(template)
        self.assertTrue(self.trans_matrix_matches(tm, template))


if __name__ == '__main__':
    unittest.main()
