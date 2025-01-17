import copy
import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from clease.settings import CEBulk, Concentration, ClusterExpansionSettings, CECrystal
from clease.settings.settings import PrimitiveCellNotFound
from clease.cluster import ClusterManager, Cluster
from clease.calculator import Clease
from clease.tools import wrap_and_sort_by_position


@pytest.fixture
def dummy_eci():
    return {"c0": 0.0}


@pytest.fixture
def atoms():
    a = 3.8  # Lattice parameter
    ats = bulk("Au", crystalstructure="fcc", a=a)
    ats = ats * (3, 3, 3)
    for i in range(int(len(ats) / 2)):
        ats[i].symbol = "Au"
        ats[-i - 1].symbol = "Cu"

    ats = wrap_and_sort_by_position(ats)
    return ats


def make_au_cu_settings(db_name, **kwargs):
    a = 3.8  # Lattice parameter
    basis_elements = [["Au", "Cu"]]
    concentration = Concentration(basis_elements=basis_elements)
    params = dict(
        crystalstructure="fcc",
        a=a,
        size=[3, 3, 3],
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[5.0, 5.0],
    )
    params.update(kwargs)

    settings = CEBulk(**params)
    return settings


@pytest.fixture
def make_settings(db_name):
    def _make_settings(**kwargs):
        return make_au_cu_settings(db_name, **kwargs)

    return _make_settings


def make_TaO(db_name, background=True):
    basis = [
        (0.0, 0.0, 0.0),
        (0.3894, 0.1405, 0.0),
        (0.201, 0.3461, 0.5),
        (0.2244, 0.3821, 0.0),
    ]
    spacegroup = 55
    cellpar = [6.25, 7.4, 3.83, 90, 90, 90]
    size = [2, 2, 2]
    basis_elements = [["O", "X"], ["O", "X"], ["O", "X"], ["Ta"]]
    grouped_basis = [[0, 1, 2], [3]]
    concentration = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    settings = CECrystal(
        basis=basis,
        spacegroup=spacegroup,
        cellpar=cellpar,
        size=size,
        concentration=concentration,
        db_name=db_name,
        max_cluster_dia=[4.0, 4.0],
        include_background_atoms=background,
    )
    return settings


@pytest.fixture
def settings_and_atoms(atoms, make_settings):
    """Create some bulk settings, and a matching AuCu atoms bulk cell"""
    return make_settings(), atoms


@pytest.mark.parametrize(
    "settings_maker",
    [
        make_au_cu_settings,
        lambda db_name: make_TaO(db_name, background=True),
        lambda db_name: make_TaO(db_name, background=False),
    ],
)
def test_prim_ordering(db_name, settings_maker):
    settings = settings_maker(db_name)
    assert all(a.index == a.tag for a in settings.prim_cell)
    # Check all tags are in order
    tags = [a.tag for a in settings.prim_cell]
    assert tags == sorted(tags)

    # Check the generator primitive cell is also in order
    prim = settings.cluster_mng.generator.prim
    assert all(a.index == a.tag for a in prim)


@pytest.mark.parametrize(
    "settings_maker",
    [
        make_au_cu_settings,
        lambda db_name: make_TaO(db_name, background=True),
        lambda db_name: make_TaO(db_name, background=False),
    ],
)
def test_get_cluster_from_cf_name(settings_maker, db_name):
    settings: ClusterExpansionSettings = settings_maker(db_name)

    for name in settings.all_cf_names:
        cluster = settings.get_cluster_corresponding_to_cf_name(name)
        assert isinstance(cluster, Cluster)
        if cluster.size > 1:
            assert isinstance(cluster.diameter, float)
        assert name.startswith(cluster.name)
    with pytest.raises(ValueError):
        settings.get_cluster_corresponding_to_cf_name("foo")
    with pytest.raises(RuntimeError):
        settings.get_cluster_corresponding_to_cf_name("c10_d0000")


def test_TaO_no_bkg(db_name):
    """The TaO settings with backgrounds has Ta as background element.
    Test the manager in the settings properly finds that."""
    settings = make_TaO(db_name, background=False)

    gen_prim = settings.cluster_mng.generator.prim
    prim = settings.prim_cell

    assert len(prim) == 14
    assert len(gen_prim) == 10
    is_bkg_all = settings.cluster_mng.is_background_atoms(prim)
    for atom in prim:
        is_bkg = atom.symbol == "Ta"
        assert is_bkg_all[atom.index] == is_bkg


def test_get_figures_settings(settings_and_atoms, dummy_eci):
    """Regression test, see issue #263.

    After getting figures from the settings' cluster manager,
    attaching a calculator would result in a RuntimeError.
    """
    settings, atoms = settings_and_atoms
    eci = dummy_eci

    # This part crashed in #263
    settings.cluster_mng.get_figures()
    calc = Clease(settings, eci)
    atoms.calc = calc
    assert atoms.calc is not None
    assert isinstance(atoms.calc, Clease)


def test_initialization_trans_matrix_cluster_list(mocker, make_settings):
    """Test we can initialize settings without constructing the cluster list and trans
    matrix until requested"""
    mocker.spy(ClusterExpansionSettings, "create_cluster_list_and_trans_matrix")

    settings = make_settings()
    # We should not have triggered any of the cached properties to be calculated yet
    assert settings.create_cluster_list_and_trans_matrix.call_count == 0
    assert settings._trans_matrix is None
    assert settings._cluster_list is None
    tm = settings.trans_matrix  # Trigger a calculation
    assert settings.create_cluster_list_and_trans_matrix.call_count == 1
    assert settings._trans_matrix is not None
    assert settings._trans_matrix is tm
    assert settings._cluster_list is not None
    # We should be able to re-call the trans matrix, without retriggering a new call
    settings.trans_matrix
    assert settings.create_cluster_list_and_trans_matrix.call_count == 1
    settings.clear_cache()
    assert settings._trans_matrix is None
    assert settings._cluster_list is None
    assert settings.create_cluster_list_and_trans_matrix.call_count == 1
    settings.trans_matrix
    assert settings.create_cluster_list_and_trans_matrix.call_count == 2
    assert settings._trans_matrix is not None
    assert settings._cluster_list is not None


def test_initialization_settings_cluster_mng(mocker, make_settings):
    """Test that we do not build the clusters in the ClusterManager
    when initializing the settings.
    """
    mocker.spy(ClusterManager, "build")

    settings = make_settings()
    assert settings.cluster_mng.build.call_count == 0
    settings.create_cluster_list_and_trans_matrix()
    assert settings.cluster_mng.build.call_count == 1

    settings.clear_cache()
    assert settings.cluster_mng.build.call_count == 1

    settings.create_cluster_list_and_trans_matrix()
    assert settings.cluster_mng.build.call_count == 2


def test_cluster_table(make_settings):
    settings = make_settings()
    res = settings.clusters_table()
    assert isinstance(res, str)
    print(res)


def test_ensure_clusters_exist(make_settings):
    settings = make_settings()
    # No cluster list should exist yet
    assert settings._cluster_list is None
    settings.ensure_clusters_exist()
    # We should've now triggered the cluster list creation
    assert settings._cluster_list is not None
    # Ensure a subsequential call has no effect
    lst = settings._cluster_list
    cpy = copy.deepcopy(lst)
    settings.ensure_clusters_exist()
    assert settings._cluster_list is lst
    assert lst == cpy


def test_get_all_figures(make_settings):
    settings = make_settings()
    figures = settings.get_all_figures_as_atoms()
    assert all(isinstance(atoms, Atoms) for atoms in figures)


def test_get_prim_cell_id(make_settings, make_tempfile):
    db2 = make_tempfile("another_database.db")
    settings = make_settings()
    # The constructor should've added the primitive cell, no error
    settings.get_prim_cell_id()
    settings._db_name = db2
    with pytest.raises(PrimitiveCellNotFound):
        settings.get_prim_cell_id()
    uid1 = settings.get_prim_cell_id(write_if_missing=True)
    # Now the cell should be there, no more error
    uid2 = settings.get_prim_cell_id()
    assert uid1 == uid2


def test_db_name(make_settings, make_tempfile):
    db2 = make_tempfile("another_database.db")
    settings = make_settings()
    # The constructor should've added the primitive cell, no error
    settings.get_prim_cell_id()
    # Set the underlying db_name, without calling the setter
    settings._db_name = db2
    with pytest.raises(PrimitiveCellNotFound):
        settings.get_prim_cell_id()
    settings.db_name = db2
    # Now the primitive should be there.
    settings.get_prim_cell_id()


def test_prim_wrap():
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    prim = bulk("Au", crystalstructure="fcc", a=4.0)
    # shift the x-lattice vector, ensure the site is outside of the cell
    prim[0].x -= prim.cell.cellpar()[0]

    settings = ClusterExpansionSettings(prim, conc)
    assert settings.prim_cell is not prim
    assert not np.allclose(settings.prim_cell.positions, prim.positions)
    prim.wrap()
    assert np.allclose(settings.prim_cell.positions, prim.positions)


def test_set_size_and_supercell_Factor(make_settings):
    settings = make_settings()

    settings.size = [3, 3, 3]
    # Test the size is the same address in memory.
    assert settings.size is settings.template_atoms.size
    assert (settings.size == np.diag([3, 3, 3])).all()
    assert settings.supercell_factor is None

    settings.supercell_factor = 28
    assert settings.size is None
    assert settings.template_atoms.size is None
    assert settings.supercell_factor == 28
    assert settings.template_atoms.supercell_factor == 28


@pytest.mark.parametrize("value", [True, False])
def test_set_background_setter(make_settings, value):
    settings = make_settings()
    with pytest.raises(NotImplementedError):
        settings.include_background_atoms = value


def test_change_max_cluster_dia(make_settings):
    settings: ClusterExpansionSettings = make_settings()
    assert settings.requires_build()
    settings.ensure_clusters_exist()
    assert not settings.requires_build()
    settings.max_cluster_dia = [3, 3, 3]
    assert settings.requires_build()
    assert isinstance(settings.max_cluster_dia, np.ndarray)
    assert np.allclose(settings.max_cluster_dia, [3, 3, 3])
