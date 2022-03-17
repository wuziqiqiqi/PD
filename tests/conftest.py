import os
from pathlib import Path
import random
import copy
import shutil

import pytest
import numpy as np
import ase
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


@pytest.fixture(autouse=True)
def auto_tmp_workdir(tmp_path):
    """Fixture to automatically switch working directory of the test
    to the tmp path.
    Will automatically be used, so it does not need to be used explicitly
    by the test.
    """
    path = Path(str(tmp_path))  # Normalize path
    with ase.utils.workdir(path, mkdir=True):
        yield


def remove_file(name):
    """Helper function for removing files"""
    try:
        # Pytest path-objects raises a different error if the file
        # does not exist, so we always just use os.remove
        os.remove(name)
    except FileNotFoundError:
        pass
    assert not name.exists(), f"File {name} still exists after teardown."


@pytest.fixture
def make_tempfile(tmpdir):
    """Factory function for creating temporary files.
    The file will be removed at teardown of the fixture."""

    created_files = []  # Keep track of files which are created

    def _make_tempfile(filename):
        name = tmpdir / filename
        assert not name.exists(), f"File {name} already exists."
        created_files.append(name)
        return str(name)

    yield _make_tempfile
    # Teardown
    for name in created_files:
        # Note: The file does not necessarily exist, just because we created the filename
        remove_file(name)


@pytest.fixture
def db_name(make_tempfile):
    """Create a temporary database file"""
    yield make_tempfile("temp_db.db")


@pytest.fixture
def traj_file(make_tempfile):
    """Create a temporary trajectory file"""
    yield make_tempfile("temp_trajectory.traj")


@pytest.fixture
def buffer_file(make_tempfile):
    yield make_tempfile("temp_buffer.txt")


@pytest.fixture
def make_conc():
    def _make_conc(basis_elements, **kwargs):
        return Concentration(basis_elements=basis_elements, **kwargs)

    return _make_conc


@pytest.fixture(scope="module")
def make_module_tempfile(tmpdir_factory):
    """Same fixture as make_tempfile, but scoped to the module level"""
    created_files = []  # Keep track of files which are created

    def _make_module_tempfile(filename, folder="evaluate"):
        name = tmpdir_factory.mktemp(folder).join(filename)
        assert not name.exists()
        created_files.append(name)
        return str(name)

    yield _make_module_tempfile
    for file in created_files:
        remove_file(file)
        assert not file.exists()


# This takes a few seconds to create every time, so we scope it to the module level
# No modifications should be made to this DB though, as changes will propagate throughout the test


@pytest.fixture(scope="module")
def module_bc_setting(make_module_tempfile):
    db_name = make_module_tempfile("module_temp_db.db")
    basis_elements = [["Au", "Cu"]]
    conc = Concentration(basis_elements=basis_elements)
    settings = CEBulk(
        concentration=conc,
        crystalstructure="fcc",
        a=4.05,
        max_cluster_dia=[5, 5, 5],
        size=[3, 3, 3],
        db_name=db_name,
    )
    newstruct = NewStructures(settings, struct_per_gen=3)
    newstruct.generate_initial_pool()
    calc = EMT()

    with connect(db_name) as database:
        for row in database.select([("converged", "=", False)]):
            atoms = row.toatoms()
            atoms.calc = calc
            atoms.get_potential_energy()
            update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    return settings


@pytest.fixture()
def bc_setting(make_tempfile, module_bc_setting):
    """A default test setup with AuCu. Makes a copy of the module-wide
    fixture."""
    settings = copy.deepcopy(module_bc_setting)
    session_temp_db = make_tempfile("session_temp_db.db")
    settings.db_name = session_temp_db

    # Make a copy of the old database, so we do not alter the original
    # but the test is allowed to alter this database however it sees fit.
    shutil.copyfile(module_bc_setting.db_name, session_temp_db)
    return settings


@pytest.fixture
def compare_dict():
    """Fixture for comparing dictionaries that should be equal."""

    def _compare_dict(dct1, dct2):
        assert isinstance(dct1, dict)
        assert isinstance(dct2, dict)

        assert dct1.keys() == dct2.keys()

        for key in dct1:
            val1 = dct1[key]
            val2 = dct2[key]
            assert type(val1) is type(val2)
            if isinstance(val1, (np.ndarray)):
                assert np.allclose(val1, val2)
            elif isinstance(val1, dict):
                # Recursively unpack the dictionary
                _compare_dict(val1, val2)
            else:
                assert val1 == val2, type(val1)

    return _compare_dict


@pytest.fixture
def set_rng():
    """Fixture to set the seed of random number generators"""

    def _set_rng(seed=42):
        random.seed(seed)

    return _set_rng


@pytest.fixture(autouse=True)
def default_seed(set_rng):
    """
    Automatically set the random seed before a test.
    """
    set_rng()


@pytest.fixture
def compare_atoms():
    def _compare_atoms(atoms1, atoms2):
        if len(atoms1) != len(atoms2):
            return False

        if not np.allclose(atoms1.cell, atoms2.cell):
            return False

        if atoms1.arrays.keys() != atoms2.arrays.keys():
            return False

        for k in atoms1.arrays:
            # positions, numbers, tags, etc...
            arr1 = atoms1.arrays[k]
            arr2 = atoms2.arrays[k]
            if not np.allclose(arr1, arr2):
                return False
        return True

    return _compare_atoms
