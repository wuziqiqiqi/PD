import itertools
import pytest

from clease_cxx import CppCluster
from clease.cluster import Cluster, ClusterFingerprint


@pytest.fixture
def cluster():
    true_fp = ClusterFingerprint([4.5, 4.3, 2.4, -1.0, -3.4, -1.0])
    true_indices = [[0, 3, 3], [1, 0, 5]]
    true_equiv_sites = [[0, 1]]
    cluster = Cluster(name='c3_d0001_0',
                      size=3,
                      diameter=5.4,
                      fingerprint=true_fp,
                      ref_indx=0,
                      indices=true_indices,
                      equiv_sites=true_equiv_sites,
                      trans_symm_group=0)
    return cluster


def test_get_size(cluster):

    cpp_cluster = CppCluster(cluster)

    exp_size = cluster.size

    cpp_size = cpp_cluster.get_size()
    assert exp_size == cpp_size


@pytest.mark.parametrize('n_basis', [2, 3, 4, 5, 6])
def test_get_all_decoration_numbers(n_basis, cluster):
    """Test calculating all decoration numbers in the C++ Cluster class"""
    cpp_cluster = CppCluster(cluster)

    cluster_size = cluster.size
    all_deco = sorted(cpp_cluster.get_all_decoration_numbers(n_basis))

    assert len(all_deco) == n_basis**cluster_size

    # We expect the output to be comparable to an itertools product or repeat = cluster_size
    # i.e., product([0, 1], 2) -> [[0, 0], [0, 1], [1, 0], [1, 1]]
    # Since we want all possible decoration number combinations
    expected = itertools.product(range(n_basis), repeat=cluster_size)
    # Make expected a list of list, so we can do a direct comparison
    expected = [list(e) for e in expected]

    assert len(all_deco) == len(expected)
    assert all_deco == expected
