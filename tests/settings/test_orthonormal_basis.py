"""Test to ensure the orthnormality of the basis functions."""
import itertools
import pytest
from clease.settings import CEBulk

tol = 1E-9


def check_orthonormal(settings):
    """Check orthonormality."""
    for bf in settings.basis_functions:
        sum_ = 0
        for key in settings.spin_dict.keys():
            sum_ += bf[key]**2
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


bfs = ['polynomial', 'trigonometric']
bf_ids = [str(bf) for bf in bfs]
all_basis_elements = [[['Au', 'Cu']], [['Au', 'Cu', 'Ag']], [['Au', 'Cu', 'Ag', 'Ni']],
                      [['Au', 'Cu', 'Ag', 'Ni', 'Fe']], [['Au', 'Cu', 'Ag', 'Ni', 'Fe', 'H']]]
# Let's make some reasonable names for the tests
elements_ids = ['_'.join(elements[0]) for elements in all_basis_elements]


# Test all combinations of bfs and all_basis_elements
@pytest.mark.parametrize('bf', bfs, ids=bf_ids)
@pytest.mark.parametrize('basis_elements', all_basis_elements, ids=elements_ids)
def test_orthonormal(bf, basis_elements, db_name, make_conc):
    concentration = make_conc(basis_elements)
    settings = CEBulk(crystalstructure="fcc",
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      max_cluster_size=2,
                      db_name=db_name)
    settings.basis_func_type = bf
    check_orthonormal(settings)
