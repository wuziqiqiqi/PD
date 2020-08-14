import unittest
from ase.build import bulk
from clease.cluster import ClusterManager
from clease.tools import wrap_and_sort_by_position
import numpy as np


class TestClusterManager(unittest.TestCase):

    def trans_matrix_matches(self, tm, template):
        ref_indices = {}
        for row, atom in zip(tm, template):
            if all(k == v for k, v in row.items()):
                ref_indices[atom.tag] = atom.index

        for row, atom in zip(tm, template):
            for k, v in row.items():
                d_orig = template.get_distance(ref_indices[atom.tag], int(k), mic=True, vector=True)
                d = template.get_distance(atom.index, v, vector=True, mic=True)
                if not np.allclose(d, d_orig):
                    return False
        return True

    def test_tm_fcc(self):
        prim = bulk('Al')
        prim[0].tag = 0
        manager = ClusterManager(prim)
        template = prim * (3, 3, 3)
        template.wrap()
        manager.build(max_size=2, max_cluster_dia=3.0)
        trans_mat = manager.translation_matrix(template)
        self.assertTrue(self.trans_matrix_matches(trans_mat, template))

    def test_tm_hcp(self):
        prim = bulk('Mg', crystalstructure='hcp', a=3.8, c=4.8)
        prim.wrap()
        for atom in prim:
            atom.tag = atom.index
        template = prim * (4, 4, 4)
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

        template = prim * (2, 2, 2)
        manager = ClusterManager(prim)
        manager.build(max_size=2, max_cluster_dia=3.0)
        tm = manager.translation_matrix(template)
        self.assertTrue(self.trans_matrix_matches(tm, template))

    def test_lut(self):
        prim = bulk('LiO', crystalstructure='rocksalt', a=4.05)
        prim.wrap()
        for atom in prim:
            atom.tag = atom.index

        tests = [{
            'atoms': prim,
            'expect': {
                (0, 0, 0, 0): 0,
                (0, 0, 0, 1): 1
            }
        }, {
            'atoms': wrap_and_sort_by_position(prim * (2, 2, 2)),
            'expect': {
                (0, 0, 0, 0): 0,
                (0, 0, 0, 1): 4,
                (0, 1, 0, 0): 2,
                (0, 1, 0, 1): 9,
                (0, 0, 1, 0): 3,
                (0, 0, 1, 1): 10,
                (0, 1, 1, 0): 8,
                (0, 1, 1, 1): 14,
                (1, 0, 0, 0): 1,
                (1, 0, 0, 1): 7,
                (1, 0, 1, 0): 6,
                (1, 0, 1, 1): 13,
                (1, 1, 0, 0): 5,
                (1, 1, 0, 1): 12,
                (1, 1, 1, 0): 11,
                (1, 1, 1, 1): 15
            }
        }]

        manager = ClusterManager(prim)
        for i, test in enumerate(tests):
            lut = manager.create_four_vector_lut(test['atoms'])
            msg = 'Test #{} failed.\nGot: {}\nExpected: {}'.format(i, lut, test['expect'])
            self.assertDictEqual(lut, test['expect'], msg=msg)


if __name__ == '__main__':
    unittest.main()
