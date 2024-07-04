import os
import pytest
from ase.build import bulk
from clease.settings import CEBulk, Concentration
from clease import NewStructures
from clease.tools import wrap_and_sort_by_position


def test_formula_unit(db_name):
    basis_elements = [["Li", "Ru", "X"], ["O", "X"]]
    A_eq = [[0, 3, 0, 0, 0]]
    b_eq = [1]
    A_lb = [[0, 0, 0, 3, 0]]
    b_lb = [2]
    conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq, A_lb=A_lb, b_lb=b_lb)
    settings = CEBulk(
        crystalstructure="rocksalt",
        a=4.0,
        size=[2, 2, 3],
        concentration=conc,
        db_name=db_name,
        max_cluster_dia=[4.0],
    )
    newstruct = NewStructures(settings=settings)

    test = bulk(name="LiO", crystalstructure="rocksalt", a=4.0) * (2, 2, 3)
    atoms = wrap_and_sort_by_position(test.copy())
    atoms[0].symbol = "Ru"
    atoms[1].symbol = "X"
    atoms[4].symbol = "X"
    fu = newstruct._get_formula_unit(atoms)
    assert fu == "Li10Ru1X1_O11X1"

    atoms = wrap_and_sort_by_position(test.copy())
    fu = newstruct._get_formula_unit(atoms)
    assert fu == "Li1_O1"

    atoms = wrap_and_sort_by_position(test.copy())
    replace_cat = [0, 1, 2, 5, 6, 14]
    replace_an = [4, 7, 9, 13, 15, 20]
    for cat in replace_cat:
        atoms[cat].symbol = "X"
    for an in replace_an:
        atoms[an].symbol = "X"

    fu = newstruct._get_formula_unit(atoms)
    assert fu == "Li1X1_O1X1"
