"""Unit tests for the corr function class."""
import os
from clease import CEBulk, CorrFunction, Concentration
import unittest


basis_elements = [["Au", "Cu", "Si"]]
concentration = Concentration(basis_elements=basis_elements)


def get_bc_setting(db_name):
    return CEBulk(crystalstructure="fcc", a=4.05, size=[4, 4, 4],
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
        db_name = "test_corrfunc_transmat.db"
        bc_setting = get_bc_setting(db_name)
        atoms = bc_setting.atoms
        tm = bc_setting.trans_matrix
        ref_dist = atoms.get_distance(0, 1, mic=True)
        for indx in range(len(atoms)):
            dist = atoms.get_distance(indx, tm[indx][1], mic=True)
            self.assertAlmostEqual(dist, ref_dist)
        os.remove(db_name)

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
        corr.get_cf_by_names(atoms, ['c3_d0000_0_000'])

        # Try a quadruplet: Have to raise error
        with self.assertRaises(ClusterNotTrackedError):
            corr.get_cf_by_names(atoms, ['c4_d0001_0_0000'])

    def tearDown(self):
        try:
            os.remove(db_name)
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
