import pytest
from ase.build import bulk
from clease.cluster import ClusterManager
from clease.tools import wrap_and_sort_by_position
import numpy as np


@pytest.fixture
def bulk_al():
    return bulk('Al')


@pytest.fixture
def cluster_mng(bulk_al):
    return ClusterManager(bulk_al)


def trans_matrix_matches(tm, template):
    ref_indices = {}
    for row, atom in zip(tm, template):
        if all(k == v for k, v in row.items()):
            ref_indices[atom.tag] = atom.index

    for row, atom in zip(tm, template):
        for k, v in row.items():
            d_orig = template.get_distance(ref_indices[atom.tag], int(k), mic=True, vector=True)
            d = template.get_distance(atom.index, v, vector=True, mic=True)
            if not np.allclose(d, d_orig):
                return False
    return True


def test_tm_fcc():
    prim = bulk('Al')
    prim[0].tag = 0
    manager = ClusterManager(prim)
    template = prim * (3, 3, 3)
    template.wrap()
    manager.build(max_cluster_dia=[3.0])
    trans_mat = manager.translation_matrix(template)
    assert trans_matrix_matches(trans_mat, template)


def test_tm_hcp():
    prim = bulk('Mg', crystalstructure='hcp', a=3.8, c=4.8)
    prim.wrap()
    for atom in prim:
        atom.tag = atom.index
    template = prim * (4, 4, 4)
    template.wrap()
    manager = ClusterManager(prim)
    manager.build(max_cluster_dia=[4.0])
    tm = manager.translation_matrix(template)
    assert trans_matrix_matches(tm, template)


def test_tm_rocksalt():
    prim = bulk('LiX', crystalstructure='rocksalt', a=4.0)
    prim.wrap()
    for atom in prim:
        atom.tag = atom.index

    template = prim * (2, 2, 2)
    manager = ClusterManager(prim)
    manager.build(max_cluster_dia=[3.0])
    tm = manager.translation_matrix(template)
    assert trans_matrix_matches(tm, template)


def test_lut():
    prim = bulk('LiO', crystalstructure='rocksalt', a=4.05)
    prim.wrap()
    for atom in prim:
        atom.tag = atom.index

    tests = [{
        'atoms': prim,
        'expect': {
            (0, 0, 0, 0): 0,
            (0, 0, 0, 1): 1
        }
    }, {
        'atoms': wrap_and_sort_by_position(prim * (2, 2, 2)),
        'expect': {
            (0, 0, 0, 0): 0,
            (0, 0, 0, 1): 4,
            (0, 1, 0, 0): 2,
            (0, 1, 0, 1): 9,
            (0, 0, 1, 0): 3,
            (0, 0, 1, 1): 10,
            (0, 1, 1, 0): 8,
            (0, 1, 1, 1): 14,
            (1, 0, 0, 0): 1,
            (1, 0, 0, 1): 7,
            (1, 0, 1, 0): 6,
            (1, 0, 1, 1): 13,
            (1, 1, 0, 0): 5,
            (1, 1, 0, 1): 12,
            (1, 1, 1, 0): 11,
            (1, 1, 1, 1): 15
        }
    }]

    manager = ClusterManager(prim)
    for i, test in enumerate(tests):
        lut = manager.create_four_vector_lut(test['atoms'])
        msg = 'Test #{} failed.\nGot: {}\nExpected: {}'.format(i, lut, test['expect'])
        assert lut == test['expect'], msg


def test_get_figures_multiple_times(cluster_mng):
    """Regression test. After getting figures from settings,
    attaching a calculator would result in a RuntimeError.
    
    See issue #263
    """
    fig1 = cluster_mng.get_figures()
    fig2 = cluster_mng.get_figures()
    assert fig1 == fig2


@pytest.mark.parametrize('max_cluster_dia', [
    [5., 5.],
    (5., 5.),
    [],
    np.array([1, 2, 3]),
])
def test_cache(mocker, cluster_mng, max_cluster_dia):
    # max_size = 3
    # max_cluster_dia = [5, 5]
    mocker.spy(ClusterManager, '_prepare_new_build')
    assert cluster_mng._prepare_new_build.call_count == 0
    # No build has been performed yet, we should require a build
    assert cluster_mng.requires_build(max_cluster_dia)
    cluster_mng.build(max_cluster_dia)
    len_orig = len(cluster_mng.clusters)
    cluster_0 = cluster_mng.clusters[0]
    # Ensure we called prepare_new_build, and we no longer require a build
    assert cluster_mng._prepare_new_build.call_count == 1
    assert not cluster_mng.requires_build(max_cluster_dia)

    # Check we didn't perform a new build
    cluster_mng.build(max_cluster_dia)
    assert cluster_mng._prepare_new_build.call_count == 1
    assert len(cluster_mng.clusters) == len_orig
    # The first cluster must be the same object in memory, since
    # we didn't do anything to the cluster list
    assert cluster_0 is cluster_mng.clusters[0]

    # Build using a new set of arguments
    # max_size = 4
    max_cluster_dia = list(max_cluster_dia) + [5.]
    assert cluster_mng.requires_build(max_cluster_dia)
    cluster_mng.build(max_cluster_dia)
    assert cluster_mng._prepare_new_build.call_count == 2
    # The first cluster must be the same object in memory, since
    # we didn't do anything to the cluster list
    assert cluster_0 is not cluster_mng.clusters[0]
