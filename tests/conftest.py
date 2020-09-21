import os
import pytest
from ase.db import connect
from ase.calculators.emt import EMT
from clease.settings import CEBulk, Concentration
from clease import NewStructures
from clease.tools import update_db


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    # add @pytest.mark.slow to run slow tests
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def remove_file(name):
    """Helper function for removing files"""
    try:
        # Pytest path-objects raises a different error if the file
        # does not exist, so we always just use os.remove
        os.remove(name)
    except FileNotFoundError:
        pass


@pytest.fixture
def make_tempfile(tmpdir):
    """Factory function for creating temporary files.
    The file will be removed at teardown of the fixture. """

    created_files = []  # Keep track of files which are created

    def _make_tempfile(filename):
        name = tmpdir / filename
        assert not name.exists(), f'File {name} already exists.'
        created_files.append(name)
        return str(name)

    yield _make_tempfile
    # Teardown
    for name in created_files:
        # Note: The file does not necessarily exist, just because we created the filename
        remove_file(name)
        assert not name.exists(), f'File {name} still exists after teardown.'


@pytest.fixture
def db_name(make_tempfile):
    """Create a temporary database file"""
    yield make_tempfile('temp_db.db')


@pytest.fixture
def buffer_file(make_tempfile):
    yield make_tempfile('temp_buffer.txt')


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

    with connect(db_name) as database:
        for row in database.select([("converged", "=", False)]):
            atoms = row.toatoms()
            atoms.calc = calc
            atoms.get_potential_energy()
            update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    yield settings
    remove_file(name)
