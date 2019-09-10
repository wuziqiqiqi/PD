"""Test to ensure the orthnormality of the basis functions."""
import os
import itertools
from clease import CEBulk, Concentration
import unittest


tol = 1E-9


def test_2(basis_function, db_name):
    """Test for 2 element case."""
    basis_elements = [['Au', 'Cu']]
    concentration = Concentration(basis_elements=basis_elements)

    setting = CEBulk(crystalstructure="fcc",
                     a=4.05,
                     size=[3, 3, 3],
                     concentration=concentration,
                     max_cluster_size=2,
                     db_name=db_name,
                     basis_function=basis_function)
    check_orthonormal(setting)


def test_3(basis_function, db_name):
    """Test for 3 element case."""
    basis_elements = [['Au', 'Cu', 'Ag']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure="fcc",
                     a=4.05,
                     size=[3, 3, 3],
                     concentration=concentration,
                     max_cluster_size=2,
                     db_name=db_name,
                     basis_function=basis_function)
    check_orthonormal(setting)


def test_4(basis_function, db_name):
    """Test for 4 element case."""
    basis_elements = [['Au', 'Cu', 'Ag', 'Ni']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure="fcc",
                     a=4.05,
                     size=[3, 3, 3],
                     concentration=concentration,
                     max_cluster_size=2,
                     db_name=db_name,
                     basis_function=basis_function)
    check_orthonormal(setting)


def test_5(basis_function, db_name):
    """Test for 5 element case."""
    basis_elements = [['Au', 'Cu', 'Ag', 'Ni', 'Fe']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure="fcc",
                     a=4.05,
                     size=[3, 3, 3],
                     concentration=concentration,
                     max_cluster_size=2,
                     db_name=db_name,
                     basis_function=basis_function)
    check_orthonormal(setting)


def test_6(basis_function, db_name):
    """Test for 6 element case."""
    basis_elements = [['Au', 'Cu', 'Ag', 'Ni', 'Fe', 'H']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure="fcc",
                     a=4.05,
                     size=[3, 3, 3],
                     concentration=concentration,
                     max_cluster_size=2,
                     db_name=db_name,
                     basis_function=basis_function)
    check_orthonormal(setting)


def check_orthonormal(setting):
    """Check orthonormality."""
    for bf in setting.basis_functions:
        sum = 0
        for key, _ in setting.spin_dict.items():
            sum += bf[key] * bf[key]
        sum /= setting.num_unique_elements
        assert abs(sum - 1.0) < tol

    # Check zeros
    alpha = list(range(len(setting.basis_functions)))
    comb = list(itertools.combinations(alpha, 2))
    for c in comb:
        sum = 0
        for key, _ in setting.spin_dict.items():
            sum += setting.basis_functions[c[0]][key] \
                * setting.basis_functions[c[1]][key]
        sum /= setting.num_unique_elements
        assert abs(sum) < tol


basis_function = 'polynomial'

bfs = ['polynomial', 'trigonometric']


class TestOrthonormal(unittest.TestCase):
    def test_2(self):
        db_name = 'test2.db'
        for bf in bfs:
            test_2(bf, db_name)
        os.remove(db_name)

    def test_3(self):
        db_name = 'test3.db'
        for bf in bfs:
            test_3(bf, db_name)
        os.remove(db_name)

    def test_4(self):
        db_name = 'test4.db'
        for bf in bfs:
            test_4(bf, db_name)
        os.remove(db_name)

    def test_5(self):
        db_name = 'test5.db'
        for bf in bfs:
            test_4(bf, db_name)
        os.remove(db_name)

    def test_6(self):
        db_name = 'test6.db'
        for bf in bfs:
            test_4(bf, db_name)
        os.remove(db_name)


if __name__ == '__main__':
    unittest.main()
