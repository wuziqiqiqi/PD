"""Test case for tha GAFit"""
import pytest

import numpy as np
from ase.calculators.emt import EMT
from ase.db import connect

from cleases.regression import GAFit, SaturatedPopulationError
from cleases.settings import CEBulk, Concentration
from cleases import NewStructures, Evaluate
from cleases.tools import update_db

all_cost_funcs = ("bic", "aic", "aicc")


@pytest.fixture
def dummy_cf_matrix():
    return np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 3, 3, 3, 3]])


@pytest.fixture(params=all_cost_funcs)
def cost_func(request):
    return request.param


@pytest.fixture(scope="module")
def ga_settings(make_module_tempfile):
    db_name = make_module_tempfile("temp_db_ga.db")
    basis_elements = [["Au", "Cu"]]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        crystalstructure="fcc",
        a=4.05,
        size=[2, 2, 2],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[4.06],
    )

    newstruct = NewStructures(settings, struct_per_gen=3)
    newstruct.generate_initial_pool()

    # Insert random configurations
    atoms = settings.atoms.copy()
    symbols = ["Au", "Cu"]
    for _ in range(6):
        atoms.symbols = np.random.choice(symbols, size=len(atoms))
        newstruct.insert_structure(init_struct=atoms)

    # Compute the energy of the structures
    calc = EMT()
    with connect(db_name) as database:
        # Write the atoms to the database
        # for atoms, kvp in zip(all_atoms, key_value_pairs):
        for row in database.select([("converged", "=", False)]):
            atoms = row.toatoms()
            atoms.calc = calc
            atoms.get_potential_energy()
            update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    return settings


def test_init_from_file(cost_func, ga_settings, make_tempfile):
    backup = make_tempfile("ga_test.csv")
    evaluator = Evaluate(ga_settings)
    selector = GAFit(evaluator.cf_matrix, evaluator.e_dft, fname=backup, cost_func=cost_func)
    try:
        selector.run(gen_without_change=3, save_interval=1)
    except SaturatedPopulationError:
        pass
    individuals = selector.individuals

    # Restart from file, this time data will be loaded
    selector = GAFit(evaluator.cf_matrix, evaluator.e_dft, fname=backup, cost_func=cost_func)
    assert np.allclose(individuals, selector.individuals)


@pytest.fixture
def make_dummy_ga(dummy_cf_matrix, make_tempfile):
    default_cf_matirx = dummy_cf_matrix

    def _make_dummy_ga(cf_matrix=None, e_dft=None, fname="ga_fit.csv", **kwargs):
        cf_matrix = cf_matrix or default_cf_matirx
        e_dft = e_dft or np.arange(cf_matrix.shape[0])
        fname = make_tempfile(fname)
        return GAFit(cf_matrix, e_dft, fname=fname, **kwargs)

    return _make_dummy_ga


def test_allowed_cost_funcs(make_dummy_ga, cost_func):
    for cost_func in all_cost_funcs:
        make_dummy_ga(cost_func=cost_func)
    invalid_cost_func = f"{cost_func}_abc"
    with pytest.raises(ValueError):
        make_dummy_ga(cost_func=invalid_cost_func)


def test_pop_size(make_dummy_ga):
    # Auto
    ga = make_dummy_ga(num_individuals="auto")
    cf_matrix = ga.cf_matrix
    assert ga.pop_size == 10 * cf_matrix.shape[1]

    # Even
    ga = make_dummy_ga(num_individuals=4)
    assert ga.pop_size == 4

    # Odd
    ga = make_dummy_ga(num_individuals=5)
    assert ga.pop_size == 6
