"""Test suite for TemplateAtoms."""
import os

from ase.build import bulk
from ase.spacegroup import crystal
from ase.build import niggli_reduce
from ase.db import connect
from clease import Concentration, ValidConcentrationFilter, CEBulk
from clease.template_atoms import TemplateAtoms
from clease.template_filters import (AtomsFilter, CellFilter, SkewnessFilter,
                                     DistanceBetweenFacetsFilter,
                                     CellVectorDirectionFilter)
from clease.tools import wrap_and_sort_by_position
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
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')

        template_atoms = TemplateAtoms(
            prim_cell, supercell_factor=27, size=None,
            skew_threshold=4)
        templates = template_atoms.get_all_scaled_templates()
        ref = [[1, 1, 1], [1, 1, 2], [2, 2, 2], [2, 2, 3], [2, 2, 4],
               [2, 2, 5], [2, 3, 3], [2, 3, 4], [3, 3, 3]]

        ref = [np.diag(x).tolist() for x in ref]
        sizes = [t.info['size'] for t in templates]
        self.assertEqual(ref, sizes)

    def test_valid_concentration_filter(self):

        tests = [
            {
                'system': 'NaCl',
                'func': check_NaCl_conc
            },
            {
                'system': 'LiNiMnCoO',
                'func': lambda templ, test_suite:
                    test_suite.assertGreaterEqual(len(templ), 1)
            }
        ]

        for test in tests:
            settings = get_settings_placeholder_valid_conc_filter(
                test['system'])

            template_generator = TemplateAtoms(
                settings.atoms, supercell_factor=20,
                skew_threshold=1000000000)

            conc_filter = ValidConcentrationFilter(settings.concentration,
                                                   settings.index_by_basis)
            # Check that you cannot attach an AtomsFilter as a cell
            # filter
            with self.assertRaises(TypeError):
                template_generator.add_cell_filter(conc_filter)

            template_generator.clear_filters()
            template_generator.add_atoms_filter(conc_filter)

            templates = template_generator.get_all_scaled_templates()
            test['func'](templates, self)

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
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')

        template_atoms = TemplateAtoms(
            prim_cell, supercell_factor=27, size=None,
            skew_threshold=4)

        templates = template_atoms.get_fixed_volume_templates(
            num_prim_cells=4, num_templates=100)

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

    @patch('clease.settings_bulk.ClusterExpansionSettings.create_cluster_list_and_trans_matrix')
    def test_fixed_vol_with_conc_constraint(self, *args):
        A_eq = [[3, -2]]
        b_eq = [0]
        conc = Concentration(basis_elements=[['Au', 'Cu']],
                             A_eq=A_eq, b_eq=b_eq)

        db_name = 'test_fixed_vol_conc_constraint.db'
        settings = CEBulk(crystalstructure='fcc', a=3.8, size=[1, 1, 5],
                          db_name=db_name, max_cluster_size=2,
                          max_cluster_dia=3.0, concentration=conc,
                          supercell_factor=40)
        settings.skew_threshold = 100

        tmp = settings.template_atoms

        sizes = [4, 5, 7, 10]
        valid_size = [5, 10]
        for s in sizes:
            templates = tmp.get_fixed_volume_templates(num_prim_cells=s)

            if s in valid_size:
                self.assertGreater(len(templates), 0)
            else:
                self.assertEqual(len(templates), 0)
        os.remove(db_name)

    def test_remove_atoms_filter(self):
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc')

        template_atoms = TemplateAtoms(
            prim_cell, supercell_factor=3, size=None,
            skew_threshold=4)

        f = NumAtomsFilter(16)
        template_atoms.add_atoms_filter(f)
        self.assertEqual(len(template_atoms.atoms_filters), 1)
        template_atoms.remove_filter(f)
        self.assertEqual(len(template_atoms.atoms_filters), 0)

    def test_remove_cell_filter(self):
        template_atoms = TemplateAtoms(
            bulk("Cu"), supercell_factor=3, size=None, skew_threshold=4)

        num_cell_filters = len(template_atoms.cell_filters)
        f = DummyCellFilter()
        template_atoms.add_cell_filter(f)
        self.assertEqual(len(template_atoms.cell_filters), num_cell_filters+1)
        template_atoms.remove_filter(f)
        self.assertEqual(len(template_atoms.cell_filters), num_cell_filters)

    def test_set_skewness_threshold(self):
        template_atoms = TemplateAtoms(bulk("Cu"), skew_threshold=4)

        # Set the skewthreshold
        template_atoms.skew_threshold = 100

        # Check that the Skewness filter indeed has a value of 100
        for f in template_atoms.cell_filters:
            if isinstance(f, SkewnessFilter):
                self.assertEqual(f.ratio, 100)

    def test_cell_direction_filter(self):
        db_name = 'templates_cell_direction_filter.db'
        prim_cell = bulk("Cu", a=4.05, crystalstructure='fcc', cubic=True)
        db = connect(db_name)
        db.write(prim_cell, name='primitive_cell')

        cell_filter = CellVectorDirectionFilter(
            cell_vector=2, direction=[0, 0, 1])

        template_atoms = TemplateAtoms(
            prim_cell, supercell_factor=1,
            size=None, skew_threshold=40000)

        template_atoms.add_cell_filter(cell_filter)

        templates = template_atoms.get_fixed_volume_templates(
            num_prim_cells=5, num_templates=20)

        self.assertGreater(len(templates), 1)
        for temp in templates:
            _, _, a3 = temp.get_cell()
            self.assertTrue(np.allclose(a3[:2], [0.0, 0.0]))
        os.remove(db_name)


def get_settings_placeholder_valid_conc_filter(system):
    """
    Helper functions that initialises various dummy settings classes to be
    used together with the test_valid_conc_filter_class
    """
    settings = SettingsPlaceHolder()
    if system == 'NaCl':
        prim_cell = bulk("NaCl", crystalstructure="rocksalt", a=4.0)
        settings.atoms = prim_cell
        settings.index_by_basis = [[0], [1]]

        # Force vacancy concentration to be exactly 2/3 of the Cl
        # concentration
        A_eq = [[0, 1, -2.0]]
        b_eq = [0.0]
        settings.concentration = Concentration(
            basis_elements=[['Na'], ['Cl', 'X']], A_eq=A_eq, b_eq=b_eq)

    elif system == 'LiNiMnCoO':
        a = 2.825
        b = 2.825
        c = 13.840
        alpha = 90
        beta = 90
        gamma = 120
        spacegroup = 166
        basis_elements = [['Li'],
                          ['Ni', 'Mn', 'Co'],
                          ['O']]
        basis = [(0., 0., 0.),
                 (0., 0., 0.5),
                 (0., 0., 0.259)]

        A_eq = None
        b_eq = None

        conc = Concentration(basis_elements=basis_elements,
                             A_eq=A_eq, b_eq=b_eq)
        prim_cell = crystal(symbols=['Li', 'Ni', 'O'], basis=basis,
                            spacegroup=spacegroup,
                            cellpar=[a, b, c, alpha, beta, gamma],
                            size=[1, 1, 1], primitive_cell=True)
        prim_cell = wrap_and_sort_by_position(prim_cell)
        settings.concentration = conc

        settings.index_by_basis = [[0], [2], [1, 3]]
        settings.atoms = prim_cell
    return settings


def check_NaCl_conc(templates, test_suite):
    for atoms in templates:
        num_cl = sum(1 for atom in atoms if atom.symbol == 'Cl')
        test_suite.assertAlmostEqual(2.0*num_cl/3.0, np.round(2.0*num_cl/3.0))


if __name__ == '__main__':
    unittest.main()
