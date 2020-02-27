import unittest
from clease.basis_function import (
    BasisFunction, Polynomial, Trigonometric,
    BinaryLinear
)


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

if __name__ == '__main__':
    unittest.main()
