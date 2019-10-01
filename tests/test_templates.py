"""Test suite for TemplateAtoms."""
import os
from clease.template_atoms import TemplateAtoms
from ase.build import bulk
from ase.db import connect
from ase.build import niggli_reduce
from clease import Concentration, ValidConcentrationFilter
from clease import DistanceBetweenFacetsFilter
import numpy as np
import unittest


class SettingsPlaceHolder(object):
    """
    Dummy object that simply holds the few variables needed for the test.
    Only purpose of this is to make the test fast
    """
    atoms = None
    index_by_basis = []
    conc = None


class TestTemplates(unittest.TestCase):
    def test_fcc(self):
        db_name = 'templates_fcc.db'
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')

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
        prim_cell = bulk("Mg")
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')
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

    def test_valid_concentration_filter(self):
        prim_cell = bulk("NaCl", crystalstructure="rocksalt", a=4.0)
        settings = SettingsPlaceHolder()
        settings.atoms = prim_cell
        settings.index_by_basis = [[0], [1]]

        db_name = 'test_valid_concentration.db'
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')

        # Force vacancy concentration to be exactly 2/3 of the Cl
        # concentration
        A_eq = [[0, 1, -2.0]]
        b_eq = [0.0]
        settings.conc = Concentration(
            basis_elements=[['Na'], ['Cl', 'X']], A_eq=A_eq, b_eq=b_eq)

        template_generator = TemplateAtoms(
            db_name=db_name, supercell_factor=20,
            skew_threshold=1000000000)

        os.remove(db_name)
        conc_filter = ValidConcentrationFilter(settings)
        # Check that you cannot attach an AtomsFilter as a cell
        # filter
        with self.assertRaises(TypeError):
            template_generator.add_cell_filter(conc_filter)

        template_generator.clear_filters()
        template_generator.add_atoms_filter(conc_filter)

        templates = template_generator._generate_template_atoms()

        for atoms in templates['atoms']:
            num_cl = sum(1 for atom in atoms if atom.symbol == 'Cl')
            self.assertAlmostEqual(2.0*num_cl/3.0, np.round(2.0*num_cl/3.0))

    def test_dist_filter(self):
        f = DistanceBetweenFacetsFilter(4.0)
        cell = [[0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]
        cell = np.array(cell)
        self.assertFalse(f(cell))
        cell[0, 0] = 0.3
        self.assertTrue(f(cell))

    def test_fixed_vol(self):
        db_name = 'templates_fcc_fixed_vol.db'
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')

        template_atoms = TemplateAtoms(supercell_factor=27, size=None,
                                       skew_threshold=4,
                                       db_name=db_name)
        templates = template_atoms.get_fixed_volume_templates(
            num_prim_cells=4, num_templates=100)
        os.remove(db_name)

        # Conform that the conventional cell is present
        found_conventional = False
        conventional = [4.05, 4.05, 4.05, 90, 90, 90]
        for atoms in templates:
            niggli_reduce(atoms)
            lengths_ang = atoms.get_cell_lengths_and_angles()
            if np.allclose(lengths_ang, conventional):
                found_conventional = True
                break
        self.assertTrue(found_conventional)

if __name__ == '__main__':
    unittest.main()
