"""Test case for tha GAFit"""
import os
import pytest

import numpy as np
from numpy.random import choice
from ase.calculators.emt import EMT
from ase.db import connect

from clease.regression import GAFit, SaturatedPopulationError
from clease.settings import CEBulk, Concentration
from clease import NewStructures, Evaluate
from clease.tools import update_db


@pytest.fixture
def bc_settings(db_name):
    basis_elements = [['Au', 'Cu']]
    concentration = Concentration(basis_elements=basis_elements)
    settings = CEBulk(crystalstructure='fcc',
                      a=4.05,
                      size=[3, 3, 3],
                      concentration=concentration,
                      db_name=db_name,
                      max_cluster_size=3,
                      max_cluster_dia=[0, 0, 4.06, 4.06])

    newstruct = NewStructures(settings, struct_per_gen=3)
    newstruct.generate_initial_pool()

    # Insert 30 random configurations
    atoms = settings.atoms.copy()
    symbols = ["Au", "Cu"]
    for _ in range(6):
        for atom in atoms:
            atom.symbol = choice(symbols)
        newstruct.insert_structure(init_struct=atoms)

    # Compute the energy of the structures
    calc = EMT()
    database = connect(db_name)

    # Write the atoms to the database
    # for atoms, kvp in zip(all_atoms, key_value_pairs):
    for row in database.select([("converged", "=", False)]):
        atoms = row.toatoms()
        atoms.calc = calc
        atoms.get_potential_energy()
        update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    return settings


def test_init_from_file(bc_settings, tmpdir):
    backup = str(tmpdir / "ga_test.csv")
    evaluator = Evaluate(bc_settings)
    selector = GAFit(evaluator.cf_matrix, evaluator.e_dft, fname=backup)
    try:
        selector.run(gen_without_change=3, save_interval=1)
    except SaturatedPopulationError:
        pass
    individuals = selector.individuals

    # Restart from file, this time data will be loaded
    selector = GAFit(evaluator.cf_matrix, evaluator.e_dft, fname=backup)
    assert np.allclose(individuals, selector.individuals)
    try:
        os.remove(backup)
    except OSError:
        pass
