"""Test to ensure the orthnormality of the basis functions."""
import itertools
import os
from clease.settings import CEBulk, Concentration

tol = 1E-9


def do_test_2(basis_function, db_name):
    """Test for 2 element case."""
    basis_elements = [['Au', 'Cu']]
    concentration = Concentration(basis_elements=basis_elements)

    settings = CEBulk(crystalstructure="fcc",
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      max_cluster_size=2,
                      db_name=db_name)
    #settings.basis_func_type = basis_function
    #check_orthonormal(settings)


def do_test_3(basis_function, db_name):
    """Test for 3 element case."""
    basis_elements = [['Au', 'Cu', 'Ag']]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure="fcc",
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      max_cluster_size=2,
                      db_name=db_name)
    settings.basis_func_type = basis_function
    check_orthonormal(settings)


def do_test_4(basis_function, db_name):
    """Test for 4 element case."""
    basis_elements = [['Au', 'Cu', 'Ag', 'Ni']]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure="fcc",
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      max_cluster_size=2,
                      db_name=db_name)
    settings.basis_func_type = basis_function
    check_orthonormal(settings)


def do_test_5(basis_function, db_name):
    """Test for 5 element case."""
    basis_elements = [['Au', 'Cu', 'Ag', 'Ni', 'Fe']]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure="fcc",
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      max_cluster_size=2,
                      db_name=db_name)
    settings.basis_func_type = basis_function
    check_orthonormal(settings)


def do_test_6(basis_function, db_name):
    """Test for 6 element case."""
    basis_elements = [['Au', 'Cu', 'Ag', 'Ni', 'Fe', 'H']]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure="fcc",
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      max_cluster_size=2,
                      db_name=db_name)
    settings.basis_func_type = basis_function
    check_orthonormal(settings)


def check_orthonormal(settings):
    """Check orthonormality."""
    for bf in settings.basis_functions:
        sum_ = 0
        for key, _ in settings.spin_dict.items():
            sum_ += bf[key] * bf[key]
        sum_ /= settings.num_unique_elements
        assert abs(sum_ - 1.0) < tol

    # Check zeros
    alpha = list(range(len(settings.basis_functions)))
    comb = list(itertools.combinations(alpha, 2))
    for c in comb:
        sum_ = 0
        for key, _ in settings.spin_dict.items():
            sum_ += settings.basis_functions[c[0]][key] \
                   * settings.basis_functions[c[1]][key]
        sum_ /= settings.num_unique_elements
        assert abs(sum_) < tol


def clean_db(db_name):
    try:
        os.remove(db_name)
    except OSError:
        pass


basis_function = 'polynomial'

bfs = ['polynomial', 'trigonometric']


def test_2(db_name):
    for bf in bfs:
        do_test_2(bf, db_name)
        clean_db(db_name)


def test_3(db_name):
    for bf in bfs:
        do_test_3(bf, db_name)
        clean_db(db_name)


def test_4(db_name):
    for bf in bfs:
        do_test_4(bf, db_name)
        clean_db(db_name)


def test_5(db_name):
    for bf in bfs:
        do_test_5(bf, db_name)
        clean_db(db_name)


def test_6(db_name):
    for bf in bfs:
        do_test_6(bf, db_name)
        clean_db(db_name)
