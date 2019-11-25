from clease.cluster_generator import ClusterGenerator
from clease import CEBulk, Concentration
from clease.tools import wrap_and_sort_by_position
import unittest
from ase.build import bulk
import numpy as np
import os


class TestClusterGenerator(unittest.TestCase):
    def test_sites_cutoff_fcc(self):
        atoms = bulk('Al', a=4.05)
        generator = ClusterGenerator(atoms)
        indices = generator.sites_within_cutoff(3.0, ref_lattice=0)
        indices = list(indices)

        # For FCC there should be 12 sites within the cutoff
        self.assertEqual(len(list(indices)), 12)

        # FCC within 4.1
        indices = list(generator.sites_within_cutoff(4.1, ref_lattice=0))
        self.assertEqual(len(indices), 18)

        # FCC within 5.0
        indices = list(generator.sites_within_cutoff(5.0, ref_lattice=0))
        self.assertEqual(len(indices), 42)

    def test_sites_cutoff_bcc(self):
        a = 3.8
        atoms = bulk('Fe', a=a)
        generator = ClusterGenerator(atoms)

        # Neighbour distances
        nn = np.sqrt(3)*a/2.0
        snn = a
        indices = list(generator.sites_within_cutoff(nn+0.01, ref_lattice=0))
        self.assertEqual(len(indices), 8)
        indices = list(generator.sites_within_cutoff(snn+0.01, ref_lattice=0))
        self.assertEqual(len(indices), 14)

    def test_generate_pairs_fcc(self):
        atoms = bulk('Al', a=4.05)
        generator = ClusterGenerator(atoms)
        clusters, fps = generator.generate(2, 5.0, ref_lattice=0)
        self.assertEqual(len(clusters), 3)
        self.assertEqual(len(fps), 3)

    def test_to_atom_index_fcc(self):
        atoms = bulk('Al', a=4.05)
        generator = ClusterGenerator(atoms)
        clusters, fps = generator.generate(3, 4.1, ref_lattice=0)

        # Create a conventional unit cess
        atoms = bulk('Al', a=4.05, cubic=True)*(5, 5, 5)
        atoms = wrap_and_sort_by_position(atoms)
        int_clusters = [generator.to_atom_index(c, atoms) for c in clusters]

        # Test that cluster matches alternative algorithm
        db_name = "test_triplets_map.db"
        basis_elements = [['Al', 'Cu']]
        concentration = Concentration(basis_elements=basis_elements)
        setting = CEBulk(crystalstructure='fcc', a=4.05, size=[3, 3, 3],
                         concentration=concentration, db_name=db_name,
                         max_cluster_dia=[5.0, 5.0],
                         max_cluster_size=3)

        setting.set_active_template(atoms=atoms, generate_template=True)

        clist = setting.cluster_list
        triplets = clist.get_by_size(3)
        # Loop through clusters and compare all the clusters that has a
        # matching finger print. NOTE: definition of diameter is slightly
        # different, hence all cluster will not be present
        # Here, we confirm that both algorithm extracts the exact same indices
        # for all clusters
        for c in clist:
            try:
                i = fps.index(c.fp)
                c.indices = [sorted(x) for x in c.indices]
                for f in int_clusters[i]:
                    self.assertTrue(sorted(f) in c.indices)
            except ValueError:
                pass
        os.remove(db_name)

    def test_to_atom_index_rocksalt(self):
        db_name = 'test_cluster_generator_rocksalt.db'
        basis_elements = [['Li', 'V'], ['X', 'O']]
        concentration = Concentration(basis_elements=basis_elements)
        setting = CEBulk(crystalstructure="rocksalt",
                         a=4.0,
                         size=[2, 2, 1],
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dia=[4.01, 4.01])

        lattices = {}
        for at in setting.prim_cell:
            lattices[at.symbol] = at.tag

        atoms = bulk('LiX', crystalstructure='rocksalt', a=4.0)
        atoms.wrap()
        for atom in atoms:
            atom.tag = lattices[atom.symbol]
        generator = ClusterGenerator(atoms)
        clusters1, fp1 = generator.generate(3, 6.0, ref_lattice=0)
        clusters2, fp2 = generator.generate(3, 6.0, ref_lattice=1)

        atoms = bulk('LiX', crystalstructure='rocksalt', a=4.0)*(4, 4, 4)
        atoms = wrap_and_sort_by_position(atoms)
        setting.set_active_template(atoms=atoms, generate_template=True)
        int_cluster1 = [generator.to_atom_index(c, atoms) for c in clusters1]
        int_cluster2 = [generator.to_atom_index(c, atoms) for c in clusters2]

        c_list = setting.cluster_list.get_by_size(3)

        int_clusters = [int_cluster1, int_cluster2]
        fps = [fp1, fp2]
        for c in c_list:
            c.indices = [sorted(x) for x in c.indices]
            ref = c.ref_indx
            lattice = lattices[atoms[ref].symbol]
            try:
                i = fps[lattice].index(c.fp)
                for f in int_clusters[lattice][i]:
                    self.assertTrue(sorted(f) in c.indices)
            except ValueError:
                pass

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
        tests = [
            {
                'prim': bulk('Al'),
                'atoms': bulk('Al')*(2, 2, 2),
                'site': 4,
                'lattice': 0
            },
            {
                'prim': bulk('LiX', 'rocksalt', 4.0),
                'atoms': bulk('LiX', 'rocksalt', 4.0)*(1, 2, 3),
                'site': 4,
                'lattice': 0,
            },
            {
                'prim': bulk('LiX', 'rocksalt', 4.0),
                'atoms': bulk('LiX', 'rocksalt', 4.0)*(1, 2, 3),
                'site': 5,
                'lattice': 1,
            }
        ]

        for i, test in enumerate(tests):
            test['atoms'].wrap()
            test['prim'].wrap()
            for at in test['prim']:
                at.tag = at.index
            pos = test['atoms'][test['site']].position
            gen = ClusterGenerator(test['prim'])
            lattice = gen.get_lattice(pos)
            msg = 'Test #{} falied. Expected: {} Got {}'.format(
                i, test['lattice'], lattice)
            self.assertEqual(lattice, test['lattice'], msg=msg)


if __name__ == '__main__':
    unittest.main()
