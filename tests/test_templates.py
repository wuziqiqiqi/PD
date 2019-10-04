"""Test suite for TemplateAtoms."""
import os
from clease.template_atoms import TemplateAtoms
from ase.build import bulk
from ase.db import connect
from ase.build import niggli_reduce
from clease import Concentration, ValidConcentrationFilter
from clease.template_filters import AtomsFilter, CellFilter
from clease import DistanceBetweenFacetsFilter
from clease import Concentration, CEBulk
import numpy as np
import unittest
from unittest.mock import patch


class SettingsPlaceHolder(object):
    """
    Dummy object that simply holds the few variables needed for the test.
    Only purpose of this is to make the test fast
    """
    atoms = None
    index_by_basis = []
    Concentration = None


class NumAtomsFilter(AtomsFilter):
    def __init__(self, min_num_atoms):
        self.min_num_atoms = min_num_atoms

    def __call__(self, atoms):
        return len(atoms) > self.min_num_atoms


class DummyCellFilter(CellFilter):
    def __call__(self, cell):
        return True


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
        settings.concentration = Concentration(
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

    @patch('test_templates.CEBulk._read_data')
    @patch('test_templates.CEBulk._store_data')
    @patch('test_templates.CEBulk.create_cluster_list_and_trans_matrix')
    def test_fixed_vol_with_conc_constraint(self, *args):
        A_eq = [[3, -2]]
        b_eq = [0]
        conc = Concentration(basis_elements=[['Au', 'Cu']],
                             A_eq=A_eq, b_eq=b_eq)

        db_name = 'test_fixed_vol_conc_constraint.db'
        setting = CEBulk(crystalstructure='fcc', a=3.8, size=[1, 1, 5],
                         db_name=db_name, max_cluster_size=2,
                         max_cluster_dia=3.0, concentration=conc)

        tmp = setting.template_atoms
        v_conc = ValidConcentrationFilter(setting)
        tmp.add_atoms_filter(v_conc)

        sizes = [4, 5, 7, 10]
        valid_size = [5, 10]
        for s in sizes:
            templates = tmp.get_fixed_volume_templates(num_prim_cells=s)

            if s in valid_size:
                self.assertGreater(len(templates), 0)
            else:
                self.assertEqual(len(templates), 0)
        os.remove(db_name)

    def test_apply_filter(self):
        db_name = 'templates_fcc_apply_filter.db'
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')

        template_atoms = TemplateAtoms(supercell_factor=27, size=None,
                                       skew_threshold=4,
                                       db_name=db_name)

        num_atoms = 16
        # First confirm that we have cells with less than 16 atoms
        has_less_than = False
        for atoms in template_atoms.templates['atoms']:
            if len(atoms) < num_atoms:
                has_less_than = True
                break
        self.assertTrue(has_less_than)

        # Filter all atoms with less than 16 atoms
        template_atoms.apply_filter(NumAtomsFilter(num_atoms))
        for atoms in template_atoms.templates['atoms']:
            self.assertGreaterEqual(len(atoms), num_atoms)
        os.remove(db_name)

    def test_remove_atoms_filter(self):
        db_name = 'templates_remove_atoms_filter.db'
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')

        template_atoms = TemplateAtoms(supercell_factor=3, size=None,
                                       skew_threshold=4,
                                       db_name=db_name)

        f = NumAtomsFilter(16)
        template_atoms.add_atoms_filter(f)
        self.assertEqual(len(template_atoms.atoms_filters), 1)
        template_atoms.remove_filter(f)
        self.assertEqual(len(template_atoms.atoms_filters), 0)
        os.remove(db_name)

    def test_remove_cell_filter(self):
        db_name = 'templates_remove_atoms_filter.db'
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')

        template_atoms = TemplateAtoms(supercell_factor=3, size=None,
                                       skew_threshold=4,
                                       db_name=db_name)

        num_cell_filters = len(template_atoms.cell_filters)
        f = DummyCellFilter()
        template_atoms.add_cell_filter(f)
        self.assertEqual(len(template_atoms.cell_filters), num_cell_filters+1)
        template_atoms.remove_filter(f)
        self.assertEqual(len(template_atoms.cell_filters), num_cell_filters)
        os.remove(db_name)


if __name__ == '__main__':
    unittest.main()
