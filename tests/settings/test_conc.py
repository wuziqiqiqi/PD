from collections import OrderedDict
import pytest
from pytest import approx

import numpy as np
from cleases.settings import Concentration
from cleases.settings.concentration import InvalidConstraintError


@pytest.fixture
def testconc(make_conc):
    """Special test concentration, with a particular set of basis elements,
    for convenience"""
    basis_elements = [["Li", "Ru", "X"], ["O", "X"]]

    def _testconc(**kwargs):
        return make_conc(basis_elements, **kwargs)

    return _testconc


@pytest.fixture
def do_test_formulas(make_conc):
    """Helper fixture for testing some formulas"""

    def _do_test_formulas(basis_elements, formulas, variable_range, A_eq, b_eq, A_lb, b_lb):
        conc = make_conc(basis_elements)
        conc.set_conc_formula_unit(formulas=formulas, variable_range=variable_range)

        assert np.allclose(A_eq, conc.A_eq)
        assert np.allclose(b_eq, conc.b_eq)

        assert np.allclose(A_lb, conc.A_lb)
        assert np.allclose(b_lb, conc.b_lb)

    return _do_test_formulas


def test_full_range(testconc):
    conc = testconc().get_random_concentration()
    sum1 = np.sum(conc[:3])
    assert sum1 == approx(1)
    sum2 = np.sum(conc[3:])
    assert sum2 == approx(1)


@pytest.mark.parametrize(
    "A,b",
    [
        # Different length for A_eq and b_eq
        ([[1, 0, 0, 0, 1], [2, 3, 0, 1, 0]], [0]),
        # Wrong number of columns
        ([[1, 1]], [0]),
        ([1, 0, 0, 0, 0], [0]),
        # Wrong dimension on the b vector
        ([[1, 0, 0, 0, 0]], [[0, 1]]),
    ],
)
def test_add_usr_constraints_exceptions(A, b, testconc):
    with pytest.raises(InvalidConstraintError):
        testconc().add_usr_defined_eq_constraints(A, b)


@pytest.mark.parametrize(
    "ranges",
    [
        # Wrong number of basis
        [[(0, 1), (0, 3)]],
        # Wrong number of ranges in each basis
        [[(0, 1), (0, 0.5)], [(0, 1), (0, 1)]],
        # Wrong bounds
        [[(0, 1), (0, 0.5), (-0.5, 2.1)], [(0, 1), (0, 1)]],
        # Wrong number of bounds
        [[(0, 1), (0, 0.5), (0, 1, 0.5)], [(0, 1), (0, 1)]],
    ],
)
def test_set_conc_ranges_exceptions(ranges, testconc):
    with pytest.raises(InvalidConstraintError):
        testconc().set_conc_ranges(ranges)


@pytest.mark.parametrize(
    "kwargs",
    [
        # Formula not passed
        {"variable_range": (0, 1)},
        # Variable range not passed
        {"formulas": []},
        # Wrong number of formulas
        {"formulas": [], "variable_range": (0, 1)},
    ],
)
def test_set_conc_formula_unit_exceptions(kwargs, testconc):
    """Test various cases which should raise InvalidConstraintError"""
    with pytest.raises(InvalidConstraintError):
        testconc().set_conc_formula_unit(**kwargs)


def test_fixed_composition(make_conc):
    basis_elements = [["Li", "Ru"], ["O", "X"]]
    A_eq = [[0, 3, 0, 0], [0, 0, 0, 2]]
    b_eq = [1, 1]
    conc = make_conc(basis_elements, A_eq=A_eq, b_eq=b_eq)
    rand = conc.get_random_concentration()
    assert np.allclose(rand, np.array([2.0 / 3, 1.0 / 3, 0.5, 0.5]))


def test_fix_Ru_composition(testconc):
    A_eq = [[0, 3, 0, 0, 0]]
    b_eq = [1]
    A_lb = [[0, 0, 0, 3, 0]]
    b_lb = [2]
    conc = testconc(A_eq=A_eq, b_eq=b_eq, A_lb=A_lb, b_lb=b_lb)
    rand = conc.get_random_concentration()
    sum1 = np.sum(rand[:3])
    assert sum1 == approx(1)
    sum2 = np.sum(rand[3:])
    assert sum2 == approx(1)


def test_conc_range(testconc):
    conc = testconc()
    ranges = [[(0, 1), (1.0 / 3, 1.0 / 3), (0, 1)], [(2.0 / 3, 1), (0, 1)]]
    conc.set_conc_ranges(ranges)
    rand = conc.get_random_concentration()
    sum1 = np.sum(rand[:3])
    assert sum1 == approx(1)
    sum2 = np.sum(rand[3:])
    assert sum2 == approx(1)
    assert rand[1] == approx(1 / 3)
    assert rand[3] > 2.0 / 3


def test_formula_unit1(testconc):
    conc = testconc()
    example_str = "df<12> kd542,32"
    expected_ints = [12, 542, 32]
    assert conc._get_integers(example_str) == expected_ints


def test_formula_unit2(do_test_formulas):
    basis_elements = [["Li", "Ru", "X"], ["O", "X"]]
    formulas = ["Li<x>Ru<1>X<2-x>", "O<3-y>X<y>"]
    variable_range = OrderedDict()
    variable_range["x"] = (0, 2)
    variable_range["y"] = (0, 0.75)

    A_eq = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [0, 3, 0, 0, 0]])
    b_eq = np.array([1, 1, 1])

    A_lb = np.array([[-3, 0, 0, 0, 0], [0, 0, 0, 0, -3]])
    b_lb = np.array([-2, -0.75])

    do_test_formulas(basis_elements, formulas, variable_range, A_eq, b_eq, A_lb, b_lb)


def test_formula_unit3(do_test_formulas):
    basis_elements = [["Li", "V", "X"], ["O", "F"]]
    formulas = ["Li<x>V<1>X<2-x>", "O<2>F<1>"]
    variable_range = OrderedDict({"x": (0, 2)})

    A_eq = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [0, 3, 0, 0, 0], [0, 0, 0, 3, 0]])
    b_eq = np.array([1, 1, 1, 2])

    A_lb = np.array([[-3, 0, 0, 0, 0]])
    b_lb = np.array([-2])

    do_test_formulas(basis_elements, formulas, variable_range, A_eq, b_eq, A_lb, b_lb)


def test_formula_unit4(do_test_formulas):
    basis_elements = [["Al", "Mg", "Si"]]
    formulas = ["Al<4-4x>Mg<3x>Si<x>"]
    variable_range = OrderedDict({"x": (0, 1)})

    A_eq = np.array([[1, 1, 1], [0, 1, -3]])
    b_eq = np.array([1, 0])

    A_lb = np.array([[0, 0, -4]])
    b_lb = np.array([-1])
    do_test_formulas(basis_elements, formulas, variable_range, A_eq, b_eq, A_lb, b_lb)


def test_formula_unit5(do_test_formulas):
    basis_elements = [["Al", "Mg", "Si"]]
    formulas = ["Al<2-2x>Mg<x>Si<x>"]
    variable_range = OrderedDict({"x": (0, 1)})
    A_eq = np.array([[1, 1, 1], [0, -1, 1]])
    b_eq = np.array([1, 0])

    A_lb = np.array([[0, -2, 0]])
    b_lb = np.array([-1])

    do_test_formulas(basis_elements, formulas, variable_range, A_eq, b_eq, A_lb, b_lb)


def test_formula_unit6(do_test_formulas):
    basis_elements = [["Al", "Mg", "Si", "X"]]
    formulas = ["Al<3-x-2y>Mg<y>Si<y>X<x>"]
    variable_range = {"x": (0, 1), "y": (0, 1)}
    A_eq = np.array([[1, 1, 1, 1], [0, -1, 1, 0]])
    b_eq = np.array([1, 0])

    A_lb = np.array([[0, 0, 0, -3], [0, -3, 0, 0]])
    b_lb = np.array([-1, -1])

    do_test_formulas(basis_elements, formulas, variable_range, A_eq, b_eq, A_lb, b_lb)


@pytest.mark.parametrize(
    "basis_elements,kwargs,expect",
    [
        (
            [["Al", "Mg", "Si"], ["X", "O"], ["Ta", "Se"]],
            dict(
                A_eq=[[1, 0, 0, -1, 0, 0, 0]],
                b_eq=[0],
                A_lb=[[0, 0, 1, 0, 0, -1, 0]],
                b_lb=[0],
            ),
            1,
        ),
        (
            [["Al", "Mg", "Si"], ["X", "O"], ["Ta", "Se"]],
            dict(A_lb=[[1, 0, 0, 1, 0, -1, 0]], b_lb=[0]),
            1,
        ),
        (
            [["Al", "Mg"], ["X", "O"], ["Ta", "Se"], ["Ta", "O"]],
            dict(
                A_eq=[[1, 0, 0, 0, -1, 0, 0, 0]],
                b_eq=[0],
                A_lb=[[0, 0, 1, 0, 0, 0, 0, -1]],
                b_lb=[0],
            ),
            2,
        ),
    ],
)
def test_interlinked_basis(basis_elements, kwargs, expect, make_conc):
    conc = make_conc(basis_elements, **kwargs)
    linked = conc._linked_basis

    assert sum(1 for i, num in enumerate(linked) if num == i) == expect


@pytest.mark.parametrize(
    "kwargs",
    [
        # No kwargs
        dict(),
        # Passing A_lb and b_lb
        dict(A_lb=[[1, 1]], b_lb=[0.2]),
        # Just A_eq and b_eq
        dict(A_eq=[[1, 1]], b_eq=[0.2]),
        # All use all 4
        dict(A_lb=[[1, 1]], b_lb=[0.2], A_eq=[[1, 1]], b_eq=[0.2]),
    ],
)
def test_concentration_dict_round_trip(kwargs, compare_dict):
    """
    Test that when a concentration object is converted to a dictionary and then used
    to instantiate a new concentration that these two are the same i.e. the round trip
    conc1 -> dict -> conc2: conc1 == conc2
    """
    basis_elements = [["Au", "Cu"]]
    conc1 = Concentration(basis_elements=basis_elements, **kwargs)
    conc2 = Concentration.from_dict(conc1.todict())
    assert conc1 == conc2


def test_MoS2(make_conc):
    conc = make_conc([["S"], ["Mo", "W"]])
    conc.set_conc_formula_unit(["S<2>", "Mo<1-x>W<x>"], variable_range={"x": (0, 1)})

    # Addind S<2> should have no effect since it is already a requirement that
    # the sublattice concentrations sum to 1
    A_eq = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
    b_eq = [1.0, 1.0]
    assert np.allclose(A_eq, conc.A_eq)
    assert np.allclose(b_eq, conc.b_eq)

    # The inequality constraint specify is unnessecary so A_lb should still be empty
    assert conc.A_lb.tolist() == []
    assert conc.b_lb.tolist() == []


def test_MoScAl(make_conc):
    conc = make_conc([["Mo", "Sc"], ["Al"], ["B"]])
    conc.set_conc_formula_unit(
        formulas=["Mo<x>Sc<3-x>", "Al<1>", "B<2>"], variable_range={"x": (0, 3)}
    )

    # None of the passed constraints will have an impact. Nothing should
    # happen to the underlying equations
    A_eq = [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    b_eq = [1.0, 1.0, 1.0]
    assert np.allclose(A_eq, conc.A_eq)
    assert np.allclose(b_eq, conc.b_eq)


@pytest.mark.parametrize(
    "conc",
    [
        Concentration(basis_elements=[["Au", "Cu"]]),
        Concentration(basis_elements=[["Au", "Cu", "X"]]),
        Concentration(basis_elements=[["Au", "Cu"], ["X"]]),
        Concentration.from_dict(Concentration(basis_elements=[["Au", "Cu"]]).todict()),
    ],
)
def test_no_constraints(conc):
    # Test getting some random concentrations
    for _ in range(40):
        rnd_conc = conc.get_random_concentration()
        assert np.all(rnd_conc >= 0.0)
        assert np.all(rnd_conc <= 1.0)

    # Test extrema
    for i in range(conc.num_concs):
        max_conc = conc.get_conc_max_component(i)
        assert np.all(max_conc >= 0.0)
        assert np.all(max_conc <= 1.0)

        min_conc = conc.get_conc_min_component(i)
        assert np.all(min_conc >= 0.0)
        assert np.all(min_conc <= 1.0)


@pytest.mark.parametrize(
    "conc",
    [
        Concentration(basis_elements=[["Au", "Cu"]]),
        Concentration(basis_elements=[["Au", "Cu", "X"]]),
        Concentration(basis_elements=[["Au", "Cu"], ["X"]]),
        Concentration(
            basis_elements=[["Li", "Ru", "X"], ["O", "X"]],
            A_eq=[[0, 3, 0, 0, 0]],
            b_eq=[1],
        ),
    ],
)
def test_save_load(conc, make_tempfile, compare_dict):
    file = make_tempfile("conc.json")
    conc.save(file)
    conc2 = Concentration.load(file)
    assert conc == conc2
    compare_dict(conc.todict(), conc2.todict())
