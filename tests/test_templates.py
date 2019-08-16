"""Test suite for TemplateAtoms."""
import os
from clease.template_atoms import TemplateAtoms
from ase.build import bulk
from ase.db import connect
import numpy as np
import unittest


class TestTemplates(unittest.TestCase):
    def test_fcc(self):
        db_name = 'templates_fcc.db'
        unit_cell = bulk("Cu", a=4.05, crystalstructure='fcc')
        db = connect(db_name)
        db.write(unit_cell, name='unit_cell')

        template_atoms = TemplateAtoms(supercell_factor=27, size=None,
                                       skew_threshold=4,
                                       db_name=db_name)
        dims = template_atoms.get_size()
        ref = [[1, 1, 1], [1, 1, 2], [2, 2, 2], [2, 2, 3], [2, 2, 4],
               [2, 2, 5], [2, 3, 3], [2, 3, 4], [3, 3, 3]]

        ref = [np.diag(x).tolist() for x in ref]
        self.assertTrue(dims, ref)

        os.remove(db_name)

    def test_hcp(self):
        db_name = 'templates_hcp.db'
        unit_cell = bulk("Mg")
        db = connect(db_name)
        db.write(unit_cell, name='unit_cell')
        template_atoms = TemplateAtoms(supercell_factor=27, size=None,
                                       skew_threshold=5, db_name=db_name)
        dims = template_atoms.get_size()
        ref = [[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [1, 3, 1],
               [1, 3, 2], [1, 4, 1], [2, 2, 1], [2, 2, 2], [2, 2, 3],
               [2, 2, 4], [2, 2, 5], [2, 3, 1], [2, 3, 2], [2, 3, 3],
               [2, 3, 4], [2, 4, 1], [2, 4, 2], [2, 4, 3], [2, 5, 1],
               [2, 5, 2], [2, 6, 1], [2, 6, 2], [3, 3, 1], [3, 3, 2],
               [3, 3, 3], [3, 4, 1], [3, 4, 2], [3, 5, 1], [3, 6, 1],
               [4, 4, 1], [4, 5, 1]]

        ref = [np.diag(x).tolist() for x in ref]
        self.assertTrue(dims, ref)

        os.remove(db_name)

    def test_fixed_volume(self):
        db_name = 'templates_fixed_volume.db'
        unit_cell = bulk("Al")
        db = connect(db_name)
        db.write(unit_cell, name='unit_cell')

        template_atoms = TemplateAtoms(supercell_factor=5, size=None,
                                       db_name=db_name)

        # Create 20 templates with 2 atoms
        diag_A = [2, 1, 1]
        templates = template_atoms.get_templates_given_volume(
            diag_A=diag_A, off_diag_range=1, num_templates=20)

        vol = unit_cell.get_volume()
        os.remove(db_name)
        for template in templates:
            self.assertEqual(len(template), 2)
            self.assertAlmostEqual(2*vol, template.get_volume())


if __name__ == '__main__':
    unittest.main()
