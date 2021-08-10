import pytest
from clease.corr_func import CorrFunction
from clease.settings import CEBulk


@pytest.fixture
def get_random_eci(make_rng):
    """
    Return a set of random ECIs
    """
    rng = make_rng()

    def _get_random_eci(setting):
        cfs = CorrFunction(setting).get_cf(setting.atoms)
        ecis = {k: rng.random() for k in cfs.keys()}
        return ecis

    return _get_random_eci


@pytest.fixture
def get_LiVX(make_conc, db_name):
    """Fixture with LiVX and corresponding settings"""
    basis_elements = [['Li', 'X', 'V'], ['X', 'Li', 'V']]
    concentration = make_conc(basis_elements)

    def _get_LiVX(**kwargs):
        default_settings = dict(crystalstructure='rocksalt',
                                a=4.05,
                                size=[3, 3, 3],
                                db_name=db_name,
                                concentration=concentration,
                                max_cluster_size=3,
                                max_cluster_dia=[4.0, 4.0])
        default_settings.update(**kwargs)
        setting = CEBulk(**default_settings)
        return setting

    return _get_LiVX


@pytest.fixture
def make_dummy_settings(make_conc, db_name):
    basis_elements = [['Au', 'Cu']]
    concentration = make_conc(basis_elements)

    def _make_dummy_settings(**kwargs):
        default_settings = dict(crystalstructure='fcc',
                                a=4.0,
                                size=[1, 1, 1],
                                max_cluster_size=2,
                                max_cluster_dia=[4.0],
                                db_name=db_name,
                                concentration=concentration)
        default_settings.update(**kwargs)
        settings = CEBulk(**default_settings)
        return settings

    return _make_dummy_settings
