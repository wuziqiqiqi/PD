from collections import OrderedDict
import unittest

import numpy as np
import numpy.testing
from clease.settings import Concentration
from clease.settings.concentration import InvalidConstraintError


class TestConc(unittest.TestCase):

    def test_full_range(self):
        basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
        conc_cls = Concentration(basis_elements=basis_elements)
        conc = conc_cls.get_random_concentration()
        sum1 = np.sum(conc[:3])
        self.assertAlmostEqual(sum1, 1)
        sum2 = np.sum(conc[3:])
        self.assertAlmostEqual(sum2, 1)

        # Test exceptions
        # 1) Different length for A_eq and b_eq
        A = [[1, 0, 0, 0, 1], [2, 3, 0, 1, 0]]
        b = [0]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_eq_constraints(A, b)

        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_ineq_constraints(A, b)

        # 2) Wrong number of columns
        A = [[1, 1]]
        b = [0]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_eq_constraints(A, b)

        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_ineq_constraints(A, b)

        # 3) Wrong dimension on A matrix
        A = [1, 0, 0, 0, 0]
        b = [0]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_eq_constraints(A, b)

        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_ineq_constraints(A, b)

        # 4) Wrong dimension on the b vector
        A = [[1, 0, 0, 0, 0]]
        b = [[0, 1]]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_eq_constraints(A, b)

        with self.assertRaises(InvalidConstraintError):
            conc_cls.add_usr_defined_ineq_constraints(A, b)

        # 5) Wrong number of basis
        ranges = [[(0, 1), (0, 3)]]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.set_conc_ranges(ranges)

        # 6) Wrong number of ranges in each basis
        ranges = [[(0, 1), (0, 0.5)], [(0, 1), (0, 1)]]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.set_conc_ranges(ranges)

        # 7) Wrong bounds
        ranges = [[(0, 1), (0, 0.5), (-0.5, 2.1)], [(0, 1), (0, 1)]]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.set_conc_ranges(ranges)

        # 8) Wrong number of bounds
        ranges = [[(0, 1), (0, 0.5), (0, 1, 0.5)], [(0, 1), (0, 1)]]
        with self.assertRaises(InvalidConstraintError):
            conc_cls.set_conc_ranges(ranges)

        # 9) Formula not passed
        variable_range = {"x": (0, 1)}
        with self.assertRaises(InvalidConstraintError):
            conc_cls.set_conc_formula_unit(variable_range=variable_range)

        # 10) Variable range not passed
        formulas = []
        with self.assertRaises(InvalidConstraintError):
            conc_cls.set_conc_formula_unit(formulas=formulas)

        # 11) Wrong number of formulas
        with self.assertRaises(InvalidConstraintError):
            conc_cls.set_conc_formula_unit(formulas=formulas, variable_range=variable_range)

    def fixed_composition(self):
        basis_elements = [['Li', 'Ru'], ['O', 'X']]
        A_eq = [[0, 3, 0, 0], [0, 0, 0, 2]]
        b_eq = [1, 1]
        conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq)
        rand = conc.get_random_concentration()
        self.assertTrue(np.allclose(rand, np.array([2. / 3, 1. / 3, 0.5, 0.5])))

    def test_fix_Ru_composition(self):
        basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
        A_eq = [[0, 3, 0, 0, 0]]
        b_eq = [1]
        A_lb = [[0, 0, 0, 3, 0]]
        b_lb = [2]
        conc = Concentration(basis_elements=basis_elements,
                             A_eq=A_eq,
                             b_eq=b_eq,
                             A_lb=A_lb,
                             b_lb=b_lb)
        rand = conc.get_random_concentration()
        sum1 = np.sum(rand[:3])
        self.assertAlmostEqual(sum1, 1)
        sum2 = np.sum(rand[3:])
        self.assertAlmostEqual(sum2, 1)

    def test_conc_range(self):
        basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
        conc = Concentration(basis_elements=basis_elements)
        ranges = [[(0, 1), (1. / 3, 1. / 3), (0, 1)], [(2. / 3, 1), (0, 1)]]
        conc.set_conc_ranges(ranges)
        rand = conc.get_random_concentration()
        sum1 = np.sum(rand[:3])
        self.assertAlmostEqual(sum1, 1)
        sum2 = np.sum(rand[3:])
        self.assertAlmostEqual(sum2, 1)
        self.assertAlmostEqual(rand[1], 1. / 3)
        self.assertGreater(rand[3], 2. / 3)

    def test_formula_unit1(self):
        basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
        conc = Concentration(basis_elements=basis_elements)
        # test _get_integers() function
        example_str = "df<12> kd542,32"
        expected_ints = [12, 542, 32]
        self.assertTrue(conc._get_integers(example_str) == expected_ints)

    def test_formula_unit2(self):
        basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
        formulas = ["Li<x>Ru<1>X<2-x>", "O<3-y>X<y>"]
        var_range = OrderedDict()
        var_range["x"] = (0, 2)
        var_range["y"] = (0, 0.75)
        conc = Concentration(basis_elements=basis_elements)
        conc.set_conc_formula_unit(formulas=formulas, variable_range=var_range)
        A_eq = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [0, 3, 0, 0, 0]])
        b_eq = np.array([1, 1, 1])
        self.assertTrue(np.allclose(A_eq, conc.A_eq))
        self.assertTrue(np.allclose(b_eq, conc.b_eq))
        A_lb = np.array([[-3, 0, 0, 0, 0], [0, 0, 0, 0, -3]])
        b_lb = np.array([-2, -0.75])
        self.assertTrue(np.allclose(A_lb, conc.A_lb))
        self.assertTrue(np.allclose(b_lb, conc.b_lb))

    def test_formula_unit3(self):
        basis_elements = [['Li', 'V', 'X'], ['O', 'F']]
        formulas = ["Li<x>V<1>X<2-x>", "O<2>F<1>"]
        range = OrderedDict({"x": (0, 2)})
        conc = Concentration(basis_elements=basis_elements)
        conc.set_conc_formula_unit(formulas=formulas, variable_range=range)
        A_eq = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [0, 3, 0, 0, 0], [0, 0, 0, 3, 0]])
        b_eq = np.array([1, 1, 1, 2])

        self.assertTrue(np.allclose(A_eq, conc.A_eq))
        self.assertTrue(np.allclose(b_eq, conc.b_eq))
        A_lb = np.array([[-3, 0, 0, 0, 0]])
        b_lb = np.array([-2])
        self.assertTrue(np.allclose(A_lb, conc.A_lb))
        self.assertTrue(np.allclose(b_lb, conc.b_lb))

    def test_formula_unit4(self):
        basis_elements = [['Al', 'Mg', 'Si']]
        formulas = ['Al<4-4x>Mg<3x>Si<x>']
        range = OrderedDict({"x": (0, 1)})
        conc = Concentration(basis_elements=basis_elements)
        conc.set_conc_formula_unit(formulas=formulas, variable_range=range)
        A_eq = np.array([[1, 1, 1], [0, 1, -3]])
        b_eq = np.array([1, 0])
        self.assertTrue(np.allclose(A_eq, conc.A_eq))
        self.assertTrue(np.allclose(b_eq, conc.b_eq))
        A_lb = np.array([[0, 0, -4]])
        b_lb = np.array([-1])
        self.assertTrue(np.allclose(A_lb, conc.A_lb))
        self.assertTrue(np.allclose(b_lb, conc.b_lb))

    def test_formula_unit5(self):
        basis_elements = [['Al', 'Mg', 'Si']]
        formulas = ['Al<2-2x>Mg<x>Si<x>']
        range = OrderedDict({"x": (0, 1)})
        conc = Concentration(basis_elements=basis_elements)
        conc.set_conc_formula_unit(formulas=formulas, variable_range=range)
        A_eq = np.array([[1, 1, 1], [0, -1, 1]])
        b_eq = np.array([1, 0])
        self.assertTrue(np.allclose(A_eq, conc.A_eq))
        self.assertTrue(np.allclose(b_eq, conc.b_eq))
        A_lb = np.array([[0, -2, 0]])
        b_lb = np.array([-1])
        self.assertTrue(np.allclose(A_lb, conc.A_lb))
        self.assertTrue(np.allclose(b_lb, conc.b_lb))

    def test_formula_unit6(self):
        basis_elements = [['Al', 'Mg', 'Si', 'X']]
        formulas = ['Al<3-x-2y>Mg<y>Si<y>X<x>']
        range = {"x": (0, 1), "y": (0, 1)}
        conc = Concentration(basis_elements=basis_elements)
        conc.set_conc_formula_unit(formulas=formulas, variable_range=range)
        A_eq = np.array([[1, 1, 1, 1], [0, -1, 1, 0]])
        b_eq = np.array([1, 0])
        self.assertTrue(np.allclose(A_eq, conc.A_eq))
        self.assertTrue(np.allclose(b_eq, conc.b_eq))
        A_lb = np.array([[0, 0, 0, -3], [0, -3, 0, 0]])
        b_lb = np.array([-1, -1])
        self.assertTrue(np.allclose(A_lb, conc.A_lb))
        self.assertTrue(np.allclose(b_lb, conc.b_lb))

    def test_three_interlinked_basis(self):
        basis_elements = [["Al", "Mg", "Si"], ["X", "O"], ["Ta", "Se"]]
        A_eq = [[1, 0, 0, -1, 0, 0, 0]]
        b_eq = [0]
        A_lb = [[0, 0, 1, 0, 0, -1, 0]]
        b_lb = [0]
        conc = Concentration(basis_elements=basis_elements,
                             A_eq=A_eq,
                             b_eq=b_eq,
                             A_lb=A_lb,
                             b_lb=b_lb)
        linked = conc._linked_basis
        self.assertEqual(sum(1 for i, num in enumerate(linked) if num == i), 1)

        A_lb = [[1, 0, 0, 1, 0, -1, 0]]
        b_lb = [0]

        conc = Concentration(basis_elements=basis_elements, A_lb=A_lb, b_lb=b_lb)
        linked = conc._linked_basis
        self.assertEqual(sum(1 for i, num in enumerate(linked) if num == i), 1)

    def test_two_of_four_linked_basis(self):
        basis_elements = [["Al", "Mg"], ["X", "O"], ["Ta", "Se"], ["Ta", "O"]]
        A_eq = [[1, 0, 0, 0, -1, 0, 0, 0]]
        b_eq = [0]
        A_lb = [[0, 0, 1, 0, 0, 0, 0, -1]]
        b_lb = [0]
        conc = Concentration(basis_elements=basis_elements,
                             A_eq=A_eq,
                             b_eq=b_eq,
                             A_lb=A_lb,
                             b_lb=b_lb)
        linked = conc._linked_basis
        self.assertEqual(sum(1 for i, num in enumerate(linked) if num == i), 2)

    def test_concentration_dict_round_trip(self):
        """
        Test that when a concentration object is converted to a dictionary and then used
        to instantiate a new concentration that these two are the same i.e. the round trip
        conc1 -> dict -> conc2: conc1 == conc2
        """
        conc1 = Concentration(basis_elements=[['Au', 'Cu']])
        conc2 = Concentration.from_dict(conc1.to_dict())
        self.assertEqual(conc1, conc2)

        # Also test if A_lb and b_lb is given
        conc1 = Concentration(basis_elements=[['Au', 'Cu']], A_lb=[[1, 1]], b_lb=[0.2])
        conc2 = Concentration.from_dict(conc1.to_dict())
        self.assertEqual(conc1, conc2)

    def test_MoS2(self):
        conc = Concentration(basis_elements=[['S'], ['Mo', 'W']])
        conc.set_conc_formula_unit(["S<2>", "Mo<1-x>W<x>"], variable_range={"x": (0, 1)})

        # Addind S<2> should have no effect since it is already a requirement that
        # the sublattice concentrations sum to 1
        A_eq = [[1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0]]
        b_eq = [1.0, 1.0]
        self.assertTrue(np.allclose(A_eq, conc.A_eq))
        self.assertTrue(np.allclose(b_eq, conc.b_eq))

        # The inequality constraint specify is unnessecary so A_lb should still be empty
        self.assertListEqual(conc.A_lb.tolist(), [])
        self.assertListEqual(conc.b_lb.tolist(), [])

    def test_MoScAl(self):
        conc = Concentration(basis_elements=[['Mo', 'Sc'], ['Al'], ['B']])
        conc.set_conc_formula_unit(formulas=["Mo<x>Sc<3-x>", "Al<1>", "B<2>"],
                                   variable_range={"x": (0, 3)})

        # None of the passed constraints will have an impact. Nothing should
        # happen to the underlying equations
        A_eq = [[1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]
        b_eq = [1.0, 1.0, 1.0]
        self.assertTrue(np.allclose(A_eq, conc.A_eq))
        self.assertTrue(np.allclose(b_eq, conc.b_eq))

    def test_no_constraints(self):
        tests = [
            Concentration(basis_elements=[['Au', 'Cu']]),
            Concentration(basis_elements=[['Au', 'Cu', 'X']]),
            Concentration(basis_elements=[['Au', 'Cu'], ['X']]),
            Concentration.from_dict(
                Concentration(basis_elements=[['Au', 'Cu']]).to_dict()
            )
        ]

        for test in tests:
            rnd_conc = test.get_random_concentration()
            self.assertTrue(np.all(rnd_conc >= 0.0))
            self.assertTrue(np.all(rnd_conc <= 1.0))

            for i in range(test.num_concs):
                conc = test.get_conc_max_component(i)
                self.assertTrue(np.all(conc >= 0.0))
                self.assertTrue(np.all(conc <= 1.0))

                conc = test.get_conc_min_component(i)
                self.assertTrue(np.all(conc >= 0.0))
                self.assertTrue(np.all(conc <= 1.0))



if __name__ == '__main__':
    unittest.main()
