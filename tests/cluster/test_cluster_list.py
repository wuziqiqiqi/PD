import random
import copy
import pytest
from ase.build import bulk
import numpy as np
from cleases.cluster import Cluster, ClusterList, ClusterFingerprint
from cleases.jsonio import read_json


@pytest.fixture
def make_random_fingerprint():
    def _make_random_fingerprint(length, min=-3, max=3):
        v = [random.uniform(min, max) for _ in range(length)]
        return ClusterFingerprint(v)

    return _make_random_fingerprint


@pytest.fixture
def make_cluster():
    def _make_cluster(**kwargs):
        fp = ClusterFingerprint([3.4, 4.3, 1.2, -1.0, 2.0, 3.0])
        inputs = dict(
            name="c3_d0001_0",
            size=3,
            diameter=5.4,
            fingerprint=fp,
            ref_indx=0,
            figures=(),
            indices=[[1, 0], [3, 4]],
            equiv_sites=(),
            group=0,
        )
        inputs.update(kwargs)
        return Cluster(**inputs)

    return _make_cluster


def test_save_load_fingerprint(make_tempfile, make_random_fingerprint):
    file = make_tempfile("fp.json")
    for i in range(15):
        fp = make_random_fingerprint(i)
        fp.save(file)

        fp_loaded = read_json(file)
        assert isinstance(fp_loaded, ClusterFingerprint)
        assert fp == fp_loaded


def test_save_load_cluster(make_tempfile, make_cluster, make_random_fingerprint):
    file = make_tempfile("cluster.json")
    for i in range(10):
        cluster = make_cluster(fingerprint=make_random_fingerprint(i))
        cluster.save(file)

        cluster_loaded = read_json(file)
        assert isinstance(cluster_loaded, Cluster)
        assert cluster == cluster_loaded


def test_save_load_cluster_list(make_tempfile, make_cluster, make_random_fingerprint):
    file = make_tempfile("cluster_list.json")
    cluster_lst = ClusterList()
    for i in range(10):
        cluster = make_cluster(fingerprint=make_random_fingerprint(i))
        cluster_lst.append(cluster)
    assert cluster_lst == cluster_lst
    cluster_lst.save(file)
    loaded = read_json(file)
    assert isinstance(loaded, ClusterList)
    assert cluster_lst == loaded


def test_cluster_list_eq(make_cluster, make_random_fingerprint):
    cluster_list = ClusterList()
    for i in range(1, 15):
        cluster = make_cluster(fingerprint=make_random_fingerprint(i))
        cluster_list.append(cluster)

    cpy = copy.deepcopy(cluster_list)
    assert cpy == cluster_list
    cpy.clusters[0].fingerprint = make_random_fingerprint(18)
    assert cpy != cluster_list


def test_get_occurence_counts(make_cluster):
    cluster_list = ClusterList()
    predict_number_cluster = 5
    for i in range(predict_number_cluster):
        cluster = make_cluster(ref_indx=i)
        cluster_list.append(cluster)
    cluster_from_method = cluster_list.get_occurence_counts()
    real_number_cluster = len(cluster_from_method)
    assert real_number_cluster == predict_number_cluster


def test_parent_tracker(make_cluster):
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=4.0)
    atoms.wrap()
    for atom in atoms:
        atom.tag = atom.index
    orig_pos = atoms.get_positions()
    atoms = atoms * (7, 7, 7)

    # Find ref indices
    ref_indices = []
    for i in range(2):
        diff = atoms.get_positions() - orig_pos[i, :]
        lengths = np.sum(diff**2, axis=1)
        ref_indices.append(np.argmin(diff))

    com = np.mean(atoms.get_positions(), axis=0)
    atoms.translate(com)
    atoms.wrap()

    nn_distance = 2.0
    clist = ClusterList()
    pos = atoms.get_positions()
    for i, ref in enumerate(ref_indices):
        diff = pos - pos[ref, :]
        lengths = np.sqrt(np.sum(diff**2, axis=1))
        indices = np.nonzero(lengths < nn_distance + 0.01)[0].tolist()
        indices.remove(ref)

        new_clust = make_cluster(group=i)

        clist.append(new_clust)

    # Confirm that it the datastructure knows that the nearest neighbour
    # cluster exists in both basis
    equiv = clist.get_equivalent_clusters(clist.clusters[0])
    assert len(equiv) == 2


def test_cf_names_no_equiv_sites(make_cluster):
    prefix = "c3_d0001_0"
    cluster = make_cluster(name=prefix)
    cf_names = ClusterList.get_cf_names(cluster, 2)
    expected_suffix = ["_000", "_001", "_010", "_011", "_100", "_101", "_110", "_111"]
    expected_cf_names = [prefix + s for s in expected_suffix]
    assert cf_names == expected_cf_names


def test_cf_names_with_equiv_sites(make_cluster):
    prefix = "c3_d0001_0"

    cluster = make_cluster(name=prefix, equiv_sites=[[0, 1]])

    cf_names = ClusterList.get_cf_names(cluster, 2)
    expected_suffix = ["_000", "_001", "_010", "_011", "_110", "_111"]
    expected_cf_names = [prefix + s for s in expected_suffix]
    assert cf_names == expected_cf_names


def test_get_all_cf_names(make_cluster):
    cluster_list = ClusterList()
    fp = ClusterFingerprint([3.4, 4.3, 1.2, -1.0, 2.0, 3.0])
    prefix1 = "c3_d0001_0"
    cluster1 = make_cluster(name=prefix1, equiv_sites=[[0, 1]])
    prefix2 = "c3_d0002_0"
    cluster2 = make_cluster(name=prefix2, equiv_sites=[])

    cluster_list.append(cluster1)
    cluster_list.append(cluster2)
    all_cf_names = cluster_list.get_all_cf_names(2)

    expected_suffix1 = ["_000", "_001", "_010", "_011", "_110", "_111"]
    expected_cf_names1 = [prefix1 + s for s in expected_suffix1]
    expected_suffix2 = ["_000", "_001", "_010", "_011", "_100", "_101", "_110", "_111"]
    expected_cf_names2 = [prefix2 + s for s in expected_suffix2]

    expected_cf_names = expected_cf_names1 + expected_cf_names2
    assert all_cf_names == expected_cf_names


def test_unique_indices(make_cluster):
    cluster1 = make_cluster(indices=[[0, 3, 3], [9, 9, 9]])
    cluster2 = make_cluster(indices=[[10, 12], [2, 5]])
    cluster_list = ClusterList()
    cluster_list.append(cluster1)
    cluster_list.append(cluster2)
    indices = cluster_list.unique_indices
    indices.sort()
    expected = [0, 2, 3, 5, 9, 10, 12]
    assert indices == expected


def test_unique_indices_per_symm_group(make_cluster):
    cluster1 = make_cluster(indices=[[0, 3, 3], [9, 9, 9]], group=0)
    cluster2 = make_cluster(indices=[[10, 12], [2, 5]], group=0)
    cluster3 = make_cluster(indices=[[0, 3, 3], [9, 9, 9]], group=1)
    cluster4 = make_cluster(indices=[[10, 12], [2, 5]], group=2)

    cluster_list = ClusterList()
    cluster_list.append(cluster1)
    cluster_list.append(cluster2)
    cluster_list.append(cluster3)
    cluster_list.append(cluster4)

    indices = cluster_list.unique_indices_per_group
    indices = [sorted(x) for x in indices]
    expected = [[0, 2, 3, 5, 9, 10, 12], [0, 3, 9], [2, 5, 10, 12]]
    assert indices == expected


def test_subcluster(make_cluster):
    triplet = make_cluster(size=3, indices=[[0, 3, 5], [1, 4, 6]], group=2)
    doublet = make_cluster(size=2, indices=[[0, 3], [10, 12]], group=2)
    assert doublet.is_subcluster(triplet)
    assert triplet.is_subcluster(doublet) is False

    doublet = make_cluster(size=2, indices=[[0, 8], [10, 12]], group=2)
    assert doublet.is_subcluster(triplet) is False


def test_get_subclusters(make_cluster):
    trip = make_cluster(size=3, indices=[[0, 3, 5], [1, 4, 6]], group=2)
    d1 = make_cluster(size=2, indices=[[0, 3], [10, 12]], group=2)
    d2 = make_cluster(size=2, indices=[[0, 8], [10, 12]], group=2)
    d3 = make_cluster(size=2, indices=[[0, 5], [4, 6]], group=2)
    cluster_list = ClusterList()
    cluster_list.append(trip)
    cluster_list.append(d1)
    cluster_list.append(d2)
    cluster_list.append(d3)

    subclusters = cluster_list.get_subclusters(trip)
    expected = [d1, d3]
    assert expected == subclusters


def test_get_key(make_cluster):
    cluster = make_cluster(size=3, indices=[[3, 0, 5], [1, 4, 6]], equiv_sites=[])
    key = cluster.get_figure_key([3, 0, 5])
    assert key == "3-0-5"

    cluster.equiv_sites = [[0, 1]]
    key = cluster.get_figure_key([3, 0, 5])
    assert key == "0-3-5"

    # Try to pass a figure that mimics translated indices
    key = cluster.get_figure_key([6, 4, 0])
    assert key == "4-6-0"


def test_num_occurences(make_cluster):
    c1 = make_cluster(size=3, indices=[[3, 0, 0], [0, 3, 0], [0, 4, 5]], equiv_sites=[])
    occ_count = c1.num_fig_occurences
    expected = {"3-0-0": 1, "0-3-0": 1, "0-4-5": 1}
    for k in expected.keys():
        assert occ_count[k] == expected[k]


def test_num_occurences_equiv_sites(make_cluster):
    c1 = make_cluster(size=3, indices=[[3, 0, 0], [0, 3, 0], [0, 4, 5]], equiv_sites=[[0, 1]])
    occ_count = c1.num_fig_occurences
    expected = {"0-3-0": 2, "0-4-5": 1}
    for k in expected.keys():
        assert occ_count[k] == expected[k]


def test_corresponding_figure(make_cluster):
    c1 = make_cluster(size=4, ref_indx=0, indices=[[0, 2, 5, 1], [0, 4, 6, 7]])
    trans_matrix = [
        {0: 0, 1: 1, 2: 2, 4: 4, 5: 5, 6: 6, 7: 7},
        {0: 0, 1: 7, 2: 4, 4: 2, 5: 6, 6: 5, 7: 1},
        {0: 0, 1: 6, 2: 4, 4: 2, 5: 3, 6: 5, 7: 1},
    ]
    corresponding = c1.corresponding_figure(1, [0, 4, 6, 7], trans_matrix)
    assert corresponding == [0, 2, 5, 1]

    corresponding = c1.corresponding_figure(2, [0, 2, 5, 1], trans_matrix)
    assert corresponding == [0, 4, 6, 7]


def test_num_occ_figure(make_cluster):
    c1 = make_cluster(
        size=4,
        ref_indx=0,
        name="c4_d0000_0",
        indices=[[0, 2, 5, 1], [0, 4, 6, 7]],
        group=0,
    )
    c2 = make_cluster(
        size=4,
        ref_indx=1,
        name="c4_d0000_0",
        indices=[[1, 3, 6, 2], [0, 2, 5, 1]],
        group=1,
    )
    c3 = make_cluster(
        size=3, ref_indx=1, name="c3_d0000_0", indices=[[1, 3, 6], [1, 5, 7]], group=1
    )

    cluster_list = ClusterList()
    cluster_list.append(c1)
    cluster_list.append(c2)
    cluster_list.append(c3)

    # 0 is the reference index of symm group 0
    # 1 is the reference index of symm group 1
    di = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
    tm = [di, di, di, di, di, di, di, di]

    fig_key = "0-2-5-1"
    symm_groups = [0, 1, 0, 0, 1, 0, 1, 0, 1]
    num_fig_occ = cluster_list.num_occ_figure(fig_key, "c4_d0000_0", symm_groups, tm)
    assert num_fig_occ == 4


def test_make_names_sequential(make_cluster):
    names = [
        "c2_d0001_0",
        "c2_d0005_1",
        "c2_d0011_0",
        "c2_d0111_0",
        "c2_d1111_2",
        "c3_d0000_0",
        "c3_d0001_0",
        "c3_d0010_0",
    ]
    cluster_list = ClusterList()
    for n in names:
        s = int(n[1])
        cluster_list.append(make_cluster(size=s, name=n))

    cluster_list.make_names_sequential()

    expect_names2 = [
        "c2_d0000_0",
        "c2_d0001_0",
        "c2_d0002_0",
        "c2_d0003_0",
        "c2_d0004_0",
    ]
    n = sorted(set(c.name for c in cluster_list.get_by_size(2)))
    assert expect_names2 == n

    expect_names3 = ["c3_d0000_0", "c3_d0001_0", "c3_d0002_0"]
    n = sorted(set(c.name for c in cluster_list.get_by_size(3)))
    assert expect_names3 == n


def test_names(make_cluster):
    cluster_list = ClusterList()

    c1 = make_cluster(name="c3_d0010_0")
    cluster_list.append(c1)

    assert cluster_list.names == ["c3_d0010_0"]
    c2 = make_cluster(name="c2_d0005_1")
    cluster_list.append(c2)
    assert cluster_list.names == ["c3_d0010_0", "c2_d0005_1"]


def test_get_by_name(make_cluster):
    cluster_list = ClusterList()
    names = [
        "c2_d0001_0",
        "c2_d0005_1",
        "c2_d0011_0",
        "c2_d0111_0",
        "c2_d1111_2",
        "c3_d0000_0",
        "c3_d0001_0",
        "c3_d0010_0",
    ]
    exp_clusters = []
    for name in names:
        s = int(name[1])
        c = make_cluster(name=name, size=s)
        exp_clusters.append(c)
        cluster_list.append(c)

    res = cluster_list.get_by_name(names[0])
    assert isinstance(res, list)
    assert len(res) == 1
    res = res[0]
    assert res is exp_clusters[0]
    res = cluster_list.get_by_name(names[6])
    assert isinstance(res, list)
    assert len(res) == 1
    res = res[0]
    assert res is exp_clusters[6]
