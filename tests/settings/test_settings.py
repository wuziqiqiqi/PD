import pytest
from ase.build import bulk
from clease.settings import CEBulk, Concentration, ClusterExpansionSettings
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


@pytest.fixture
def make_settings(db_name):

    def _make_settings(**kwargs):
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

    return _make_settings


@pytest.fixture
def settings_and_atoms(atoms, make_settings):
    """Create some bulk settings, and a matching AuCu atoms bulk cell"""
    return make_settings(), atoms


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
