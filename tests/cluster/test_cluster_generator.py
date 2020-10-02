import os
import unittest

import numpy as np
from ase.build import bulk

from clease.cluster import ClusterGenerator
from clease.settings import CEBulk, Concentration
from clease.tools import wrap_and_sort_by_position


class TestClusterGenerator(unittest.TestCase):

    def test_sites_cutoff_fcc(self):
        atoms = bulk('Al', a=4.05)
        generator = ClusterGenerator(atoms)
        indices = generator.sites_within_cutoff(3.0, [0, 0, 0, 0])
        indices = list(indices)

        # For FCC there should be 12 sites within the cutoff
        self.assertEqual(len(list(indices)), 12)

        # FCC within 4.1
        indices = list(generator.sites_within_cutoff(4.1, [0, 0, 0, 0]))
        self.assertEqual(len(indices), 18)

        # FCC within 5.0
        indices = list(generator.sites_within_cutoff(5.0, [0, 0, 0, 0]))
        self.assertEqual(len(indices), 42)

    def test_sites_cutoff_bcc(self):
        a = 3.8
        atoms = bulk('Fe', a=a)
        generator = ClusterGenerator(atoms)

        # Neighbour distances
        nn = np.sqrt(3) * a / 2.0
        snn = a
        indices = list(generator.sites_within_cutoff(nn + 0.01, [0, 0, 0, 0]))
        self.assertEqual(len(indices), 8)
        indices = list(generator.sites_within_cutoff(snn + 0.01, [0, 0, 0, 0]))
        self.assertEqual(len(indices), 14)

    def test_generate_pairs_fcc(self):
        atoms = bulk('Al', a=4.05)
        generator = ClusterGenerator(atoms)
        clusters, fps = generator.generate(2, 5.0, ref_lattice=0)
        self.assertEqual(len(clusters), 3)
        self.assertEqual(len(fps), 3)

    def test_equivalent_sites(self):
        atoms = bulk('Au', a=3.8)
        generator = ClusterGenerator(atoms)

        # Test pairs
        clusters, fps = generator.generate(2, 6.0)
        for c in clusters:
            equiv = generator.equivalent_sites(c[0])
            self.assertEqual(equiv, [[0, 1]])

        # Test a triplet
        clusters, fps = generator.generate(3, 3.0)

        # For the smalles triplet all sites should be equivalent
        equiv = generator.equivalent_sites(clusters[0][0])
        self.assertEqual(equiv, [[0, 1, 2]])

    def test_get_lattice(self):
        tests = [{
            'prim': bulk('Al'),
            'atoms': bulk('Al') * (2, 2, 2),
            'site': 4,
            'lattice': 0
        }, {
            'prim': bulk('LiX', 'rocksalt', 4.0),
            'atoms': bulk('LiX', 'rocksalt', 4.0) * (1, 2, 3),
            'site': 4,
            'lattice': 0,
        }, {
            'prim': bulk('LiX', 'rocksalt', 4.0),
            'atoms': bulk('LiX', 'rocksalt', 4.0) * (1, 2, 3),
            'site': 5,
            'lattice': 1,
        }]

        for i, test in enumerate(tests):
            test['atoms'].wrap()
            test['prim'].wrap()
            for at in test['prim']:
                at.tag = at.index
            pos = test['atoms'][test['site']].position
            gen = ClusterGenerator(test['prim'])
            lattice = gen.get_lattice(pos)
            msg = 'Test #{} falied. Expected: {} Got {}'.format(i, test['lattice'], lattice)
            self.assertEqual(lattice, test['lattice'], msg=msg)

    def test_get_max_distance(self):
        a = 3.8
        atoms = bulk('Fe', a=a)
        generator = ClusterGenerator(atoms)
        generator.prim.cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        true_list = [{
            'input': [[1, 2, 3, 0], [3, 4, 1, 0]],
            'result': 3.4641
        }, {
            'input': [[1, 2, 3, 0], [2, 3, 4, 0]],
            'result': 1.7321
        }, {
            'input': [[1, 2, 3, 0], [2, 3, 4, 0], [3, 4, 1, 0]],
            'result': 3.4641
        }]
        predict_list = []
        for fig in true_list:
            predict_list.append(generator.get_max_distance(fig['input']))
        for true, predict in zip(true_list, predict_list):
            self.assertAlmostEqual(true['result'], round(predict, 4))


if __name__ == '__main__':
    unittest.main()
