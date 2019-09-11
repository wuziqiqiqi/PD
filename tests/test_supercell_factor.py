import os
from clease import CEBulk, CECrystal, Concentration
import unittest


class TestSupercelLFactor(unittest.TestCase):
    def test_fcc(self):
        db_name = 'test_sc_factor_fcc_sf.db'
        conc = Concentration(basis_elements=[['Au', 'Cu']])

        setting = CEBulk(crystalstructure='fcc', a=4.0,
                         supercell_factor=8,
                         concentration=conc, db_name=db_name,
                         max_cluster_size=4, max_cluster_dia=4.01,
                         basis_function='polynomial', skew_threshold=4,
                         ignore_background_atoms=False)

        self.assertEqual(setting.template_atoms.num_templates, 3)

        os.remove(db_name)

        setting = CEBulk(crystalstructure='fcc', a=4.01,
                         supercell_factor=8, size=[2, 2, 2],
                         concentration=conc, db_name=db_name,
                         max_cluster_size=4, max_cluster_dia=4.0,
                         basis_function='polynomial', skew_threshold=4,
                         ignore_background_atoms=False)

        self.assertEqual(setting.template_atoms.num_templates, 1)

        os.remove(db_name)

    def test_crystal(self):
        db_name = 'test_sc_factor_crystal_sf.db'
        basis_elements = [['O', 'X'], ['O', 'X'],
                          ['O', 'X'], ['Ta']]
        grouped_basis = [[0, 1, 2], [3]]
        concentration = Concentration(basis_elements=basis_elements,
                                      grouped_basis=grouped_basis)

        setting = CECrystal(basis=[(0., 0., 0.),
                                   (0.3894, 0.1405, 0.),
                                   (0.201, 0.3461, 0.5),
                                   (0.2244, 0.3821, 0.)],
                            spacegroup=55,
                            cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                            supercell_factor=10,
                            concentration=concentration,
                            db_name=db_name,
                            basis_function='binary-linear',
                            max_cluster_size=3,
                            max_cluster_dia=3.0)

        self.assertTrue(setting.template_atoms.num_templates, 20)
        os.remove(db_name)


if __name__ == '__main__':
    unittest.main()
