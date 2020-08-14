import os
import pytest
from ase.db import connect
from ase.calculators.emt import EMT
from clease.settings import CEBulk, Concentration
from clease import NewStructures
from clease.tools import update_db


@pytest.fixture
def db_name(tmpdir):
    """Create a temporary database file"""
    name = tmpdir / 'temp_db.db'
    assert not name.exists()
    yield str(name)
    # Teardown
    try:
        name.remove()
    except OSError:
        pass


# This takes a few seconds to create every time, so we scope it to the module level
# No modifications should be made to this DB though, as changes will propagate throughout the test
@pytest.fixture(scope='module')
def bc_setting(tmpdir_factory):
    name = tmpdir_factory.mktemp('evaluate').join('temp_db.db')
    assert not name.exists()
    db_name = str(name)
    print(db_name)
    basis_elements = [['Au', 'Cu']]
    conc = Concentration(basis_elements=basis_elements)
    settings = CEBulk(concentration=conc,
                      crystalstructure='fcc',
                      a=4.05,
                      size=[3, 3, 3],
                      db_name=db_name)
    newstruct = NewStructures(settings, struct_per_gen=3)
    newstruct.generate_initial_pool()
    calc = EMT()
    database = connect(db_name)

    for row in database.select([("converged", "=", False)]):
        atoms = row.toatoms()
        atoms.calc = calc
        atoms.get_potential_energy()
        update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    yield settings
    try:
        name.remove()
    except OSError:
        pass