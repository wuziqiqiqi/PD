import unittest
from clease.basis_function import (BasisFunction, Polynomial, Trigonometric, BinaryLinear)


class TestBasisFunctions(unittest.TestCase):

    def test_todict(self):
        tests = [
            {
                'bf': BasisFunction(['Au', 'Cu', 'X']),
                'expect': {
                    'name': 'generic',
                    'unique_elements': ['Au', 'Cu', 'X']
                }
            },
            {
                'bf': Polynomial(['Au', 'Cu', 'X']),
                'expect': {
                    'name': 'polynomial',
                    'unique_elements': ['Au', 'Cu', 'X']
                }
            },
            {
                'bf': Trigonometric(['Au', 'Cu', 'X']),
                'expect': {
                    'name': 'trigonometric',
                    'unique_elements': ['Au', 'Cu', 'X']
                }
            },
            {
                'bf': BinaryLinear(['Au', 'Cu', 'X']),
                'expect': {
                    'name': 'binary_linear',
                    'unique_elements': ['Au', 'Cu', 'X'],
                    'redundant_element': 'Au'
                }
            },
            {
                'bf': BinaryLinear(['Au', 'Cu', 'X'], redundant_element='X'),
                'expect': {
                    'name': 'binary_linear',
                    'unique_elements': ['Au', 'Cu', 'X'],
                    'redundant_element': 'X'
                }
            },
        ]

        for test in tests:
            dct_rep = test['bf'].todict()
            self.assertDictEqual(dct_rep, test['expect'])

    def test_customize_full_cluster_name(self):

        tests = [{
            'bf': BinaryLinear(['Au', 'Cu', 'X'], redundant_element='X'),
            'full_name': ('c4_d0012_1_1000', 'c3_d0001_4_111'),
            'ans': ('c4_d0012_1_CuAuAuAu', 'c3_d0001_4_CuCuCu')
        }, {
            'bf': BinaryLinear(['Au', 'Cu', 'Zn', 'Ag'], redundant_element='Cu'),
            'full_name': ('c4_d0001_10_1210', 'c3_d0991_10_010'),
            'ans': ('c4_d0001_10_AuZnAuAg', 'c3_d0991_10_AgAuAg')
        }, {
            'bf': Polynomial(['Au', 'Cu', 'X']),
            'full_name': ('c2_d0001_99_01', 'c4_d0991_10_0122'),
            'ans': ('c2_d0001_99_01', 'c4_d0991_10_0122')
        }]

        for test in tests:
            bf = test['bf']
            for i in range(len(test['ans'])):
                name = bf.customize_full_cluster_name(test['full_name'][i])
                self.assertEqual(name, test['ans'][i])


if __name__ == '__main__':
    unittest.main()
