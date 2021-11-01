import copy
import pytest
from ase import Atoms
from ase.build import bulk
from clease.settings import CEBulk, Concentration, ClusterExpansionSettings, CECrystal
from clease.cluster import ClusterManager
from clease.calculator import Clease
from clease.tools import wrap_and_sort_by_position


@pytest.fixture
def dummy_eci():
    return {'c0': 0.0}


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
    params = dict(crystalstructure="fcc",
                  a=a,
                  size=[3, 3, 3],
                  concentration=concentration,
                  db_name=db_name,
                  max_cluster_dia=[5.0, 5.0])
    params.update(kwargs)

    settings = CEBulk(**params)
    return settings


@pytest.fixture
def make_settings(db_name):

    def _make_settings(**kwargs):
        return make_au_cu_settings(db_name, **kwargs)

    return _make_settings


def make_TaO(db_name, background=True):
    basis = [(0., 0., 0.), (0.3894, 0.1405, 0.), (0.201, 0.3461, 0.5), (0.2244, 0.3821, 0.)]
    spacegroup = 55
    cellpar = [6.25, 7.4, 3.83, 90, 90, 90]
    size = [2, 2, 2]
    basis_elements = [['O', 'X'], ['O', 'X'], ['O', 'X'], ['Ta']]
    grouped_basis = [[0, 1, 2], [3]]
    concentration = Concentration(basis_elements=basis_elements, grouped_basis=grouped_basis)

    settings = CECrystal(basis=basis,
                         spacegroup=spacegroup,
                         cellpar=cellpar,
                         size=size,
                         concentration=concentration,
                         db_name=db_name,
                         max_cluster_dia=[4.0, 4.0])
    settings.include_background_atoms = background
    return settings


@pytest.fixture
def settings_and_atoms(atoms, make_settings):
    """Create some bulk settings, and a matching AuCu atoms bulk cell"""
    return make_settings(), atoms


@pytest.mark.parametrize('settings_maker', [
    make_au_cu_settings,
    lambda db_name: make_TaO(db_name, background=True),
    lambda db_name: make_TaO(db_name, background=False),
])
def test_prim_ordering(db_name, settings_maker):
    settings = settings_maker(db_name)
    assert all(a.index == a.tag for a in settings.prim_cell)
    # Check all tags are in order
    tags = [a.tag for a in settings.prim_cell]
    assert tags == sorted(tags)

    # Check the generator primitive cell is also in order
    prim = settings.cluster_mng.generator.prim
    assert all(a.index == a.tag for a in prim)


def test_TaO_no_bkg(db_name):
    """The TaO settings with backgrounds has Ta as background element.
    Test the manager in the settings properly finds that."""
    settings = make_TaO(db_name, background=False)

    gen_prim = settings.cluster_mng.generator.prim
    prim = settings.prim_cell

    assert len(prim) == 14
    assert len(gen_prim) == 10
    for atom in prim:
        is_bkg = atom.symbol == 'Ta'
        assert settings.cluster_mng.is_background_atom(atom) == is_bkg


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
    mocker.spy(ClusterExpansionSettings, 'create_cluster_list_and_trans_matrix')

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
    mocker.spy(ClusterManager, 'build')

    settings = make_settings()
    assert settings.cluster_mng.build.call_count == 0
    settings.create_cluster_list_and_trans_matrix()
    assert settings.cluster_mng.build.call_count == 1

    settings.clear_cache()
    assert settings.cluster_mng.build.call_count == 1

    settings.create_cluster_list_and_trans_matrix()
    assert settings.cluster_mng.build.call_count == 2


@pytest.mark.parametrize('kwargs', [
    dict(max_cluster_size=4, max_cluster_dia=3.),
    dict(max_cluster_size=2, max_cluster_dia=[4.]),
    dict(max_cluster_size=1, max_cluster_dia=[]),
    dict(max_cluster_size=1, max_cluster_dia=3)
])
def test_deprecated(make_settings, kwargs):
    """Test some deprecated kwarg combinations.
    They shouldn't fail, just raise a deprecation warning."""
    with pytest.warns(DeprecationWarning):
        make_settings(**kwargs)


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
