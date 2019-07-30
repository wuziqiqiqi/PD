"""Unit tests for the corr function class."""
import os
from clease import CEBulk, CorrFunction, Concentration
from clease.corrFunc import equivalent_deco
import unittest

db_name = "test_corrfunc.db"
basis_elements = [["Au", "Cu", "Si"]]
concentration = Concentration(basis_elements=basis_elements)


bc_setting = CEBulk(crystalstructure="fcc", a=4.05, size=[4, 4, 4],
                    concentration=concentration, db_name=db_name,
                    max_cluster_size=3, max_cluster_dia=[5.73, 5.73])


def get_mic_dists(atoms, cluster):
        """Get the MIC dist."""
        dists = []
        for indx in cluster:
            dist = atoms.get_distances(indx, cluster, mic=True)
            dists.append(dist)
        return dists


class TestCorrFunc(unittest.TestCase):
    def test_trans_matrix(self):
        """Check that the MIC distance between atoms are correct."""
        atoms = bc_setting.atoms
        tm = bc_setting.trans_matrix
        ref_dist = atoms.get_distance(0, 1, mic=True)
        for indx in range(len(atoms)):
            dist = atoms.get_distance(indx, tm[indx][1], mic=True)
            self.assertAlmostEqual(dist, ref_dist)

    def test_order_indep_ref_indx(self):
        """Check that the order of the elements are independent of the ref index.

        This does only apply for clusters with only inequivalent
        sites
        """
        for _, clst in bc_setting.cluster_info_given_size(3)[0].items():
            if clst["equiv_sites"]:
                # The cluster contains symmetrically equivalent sites
                # and then this test does not apply
                continue
            cluster = clst["indices"]
            cluster_order = clst["order"]

            init_cluster = [0] + list(cluster[0])
            init_cluster = [init_cluster[indx] for indx in cluster_order[0]]

            # Make sure that when the other indices in init_cluster are ref
            # indices, the order is the same
            for ref_indx in cluster[0]:
                found_cluster = False
                for subcluster, order in zip(cluster, cluster_order):
                    new_cluster = [ref_indx]
                    for indx in subcluster:
                        trans_indx = bc_setting.trans_matrix[ref_indx][indx]
                        new_cluster.append(trans_indx)

                    # Check if all elements are the same
                    if sorted(new_cluster) == sorted(init_cluster):
                        new_cluster = [new_cluster[indx] for indx in order]
                        found_cluster = True
                        self.assertEqual(init_cluster, new_cluster)
                self.assertTrue(found_cluster)

    def test_supercell_consistency(self):
        from clease.tools import wrap_and_sort_by_position
        basis_elements = [['Li', 'X'], ['O', 'X']]
        concentration = Concentration(basis_elements=basis_elements)
        db_name_sc = "rocksalt_sc.db"
        setting = CEBulk(crystalstructure='rocksalt',
                         a=4.05,
                         size=[1, 1, 1],
                         concentration=concentration,
                         db_name=db_name_sc,
                         max_cluster_size=3,
                         max_cluster_dia=[7.0, 4.0])
        atoms = setting.atoms.copy()
        cf = CorrFunction(setting)
        cf_dict = cf.get_cf(atoms)

        atoms = wrap_and_sort_by_position(atoms*(4, 3, 2))
        cf_dict_sc = cf.get_cf(atoms)
        for k in cf_dict_sc.keys():
            self.assertAlmostEqual(cf_dict[k], cf_dict_sc[k])
        os.remove(db_name_sc)

    def test_error_message_for_non_existent_cluster(self):
        from clease.corrFunc import ClusterNotTrackedError
        basis_elements = [['Li', 'X'], ['O', 'X']]
        concentration = Concentration(basis_elements=basis_elements)
        db_name_sc = "rocksalt_sc.db"
        setting = CEBulk(crystalstructure='rocksalt',
                         a=4.05,
                         size=[1, 1, 1],
                         concentration=concentration,
                         db_name=db_name_sc,
                         max_cluster_size=3,
                         max_cluster_dia=[7.0, 4.0])

        corr = CorrFunction(setting)
        atoms = setting.atoms
        # No error should occure
        corr.get_cf_by_cluster_names(atoms, ['c3_03nn_0_000'])

        # Try a quadruplet: Have to raise error
        with self.assertRaises(ClusterNotTrackedError):
            corr.get_cf_by_cluster_names(atoms, ['c4_01nn_0_0000'])

    def tearDown(self):
        try:
            os.remove(db_name)
        except Exception:
            pass


def time_jit():
    from clease.tools import wrap_and_sort_by_position
    import time
    basis_elements = [['Li', 'X'], ['O', 'X']]
    concentration = Concentration(basis_elements=basis_elements)
    db_name_sc = "rocksalt_sc.db"
    setting = CEBulk(crystalstructure='rocksalt',
                     a=4.05,
                     size=[1, 1, 1],
                     concentration=concentration,
                     db_name=db_name_sc,
                     max_cluster_size=3,
                     max_cluster_dia=[7.0, 4.0])
    atoms = setting.atoms.copy()
    atoms = wrap_and_sort_by_position(atoms*(4, 3, 2))

    cf = CorrFunction(setting)
    start = time.time()
    cf.get_cf(atoms)

    for n in range(10):
        start = time.time()
        cf.get_cf(atoms)
        print(time.time() - start)


if __name__ == '__main__':
    unittest.main()
