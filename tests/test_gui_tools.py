import unittest
from clease.gui.util import parse_temperature_list
from clease.gui.util import parse_concentration_list
from clease.gui.util import parse_cellpar, parse_cell
from clease.gui.util import parse_coordinate_basis
import numpy as np


class TestGUIUtil(unittest.TestCase):
    def test_temp_list(self):
        temps = parse_temperature_list('1000, 900, 500, 345.6')
        expect = [1000, 900, 500, 345.6]
        self.assertTrue(np.allclose(temps, expect))

    def test_parse_conc_list(self):
        concs = parse_concentration_list('(0.5, 0.3, 0.2), (0.1, 0.9)')
        expect = [[0.5, 0.3, 0.2], [0.1, 0.9]]

        for i in range(len(concs)):
            self.assertTrue(np.allclose(concs[i], expect[i]))

    def test_parse_conc_list_one_basis(self):
        concs = parse_concentration_list('0.5, 0.3, 0.2')
        expect = [[0.5, 0.3, 0.2]]
        self.assertTrue(np.allclose(expect[0], concs[0]))

    def test_cell_par_two_numbers(self):
        with self.assertRaises(ValueError):
            parse_cellpar('(3.0, 4.0)')

    def test_cell_lengths_and_angles(self):
        values = parse_cellpar('6.0, 7.0, 3.0, 80, 20, 10')
        expected = [6.0, 7.0, 3.0, 80, 20, 10]
        self.assertTrue(np.allclose(values, expected))

    def test_cell_par_missing_angle(self):
        with self.assertRaises(ValueError):
            parse_cellpar('6.0, 7.0, 3.0, 80, 20')

    def test_cell_par_with_parenthesis(self):
        values = parse_cellpar('(6.0, 7.0, 3.0, 80, 20, 10)')
        expected = [6.0, 7.0, 3.0, 80, 20, 10]
        self.assertTrue(np.allclose(values, expected))

    def test_cell_par_with_square_brackets(self):
        values = parse_cellpar('[6.0, 7.0, 3.0, 80, 20, 10]')
        expected = [6.0, 7.0, 3.0, 80, 20, 10]
        self.assertTrue(np.allclose(values, expected))

    def test_cell_list(self):
        values = parse_cell('[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]')
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        self.assertTrue(np.allclose(expected, values))

    def test_cell_tuple_of_lists(self):
        values = parse_cell('([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])')
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        self.assertTrue(np.allclose(expected, values))

    def test_cell_list_of_lists(self):
        values = parse_cell('[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]')
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        self.assertTrue(np.allclose(expected, values))

    def test_cell_list_of_mixed_lists_and_tuples(self):
        values = parse_cell('[ [1.0, 2.0, 3.0], (4.0, 5, 6.0), (7.0, 8, 9.0) ] ')
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        self.assertTrue(np.allclose(expected, values))

    def test_cell_nested_list_mix_tuples(self):
        values = parse_cell('[[[ [3.0, 2.0, 1.0], (4.0, 5, 6.0), (7.0, 8, 9.0) ]]] ')
        expected = [[3.0, 2.0, 1.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        self.assertTrue(np.allclose(expected, values))

    def test_crd_basis_two_tuples(self):
        values = parse_coordinate_basis('(1.0, 4.0, 4.0), (2.0, 1.0, 6.0)')
        expected = [[1.0, 4.0, 4.0], [2.0, 1.0, 6.0]]
        self.assertTrue(np.allclose(values, expected))

    def test_crd_basis_two_many_coordinates(self):
        with self.assertRaises(ValueError):
            parse_coordinate_basis('(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0, 1.0)')

    def test_crd_basis_three_lists(self):
        values = parse_coordinate_basis('[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]')
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        self.assertTrue(np.allclose(values, expected))

    def test_crd_basis_tuple_of_lists(self):
        text = '([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])'
        values = parse_coordinate_basis(text)
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        self.assertTrue(np.allclose(expected, values))

    def test_crd_basis_mix_tuple_list_missing_crd(self):
        text = '[ [1.0, 2.0, 3.0], (4.0), (7.0, 8, 9.0) ] '
        with self.assertRaises(ValueError):
            parse_coordinate_basis(text)

    def test_crd_basis_multiple_nested_lists(self):
        text = '[[[ [3.0, 2.0, 1.0] ]]] '
        values = parse_coordinate_basis(text)
        expected = [[3.0, 2.0, 1.0]]
        self.assertTrue(np.allclose(values, expected))


if __name__ == '__main__':
    unittest.main()