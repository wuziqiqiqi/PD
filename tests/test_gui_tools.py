import unittest
from clease.gui.util import parse_temperature_list
from clease.gui.util import parse_concentration_list
from clease.gui.util import parse_cellpar
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


if __name__ == '__main__':
    unittest.main()
