from collections import Counter
import pytest
from ase.build import bulk
from clease.cluster import ClusterManager
from clease.datastructures.four_vector import FourVector
from clease.tools import wrap_and_sort_by_position, get_size_from_cf_name
import numpy as np


def tagger(atoms):
    for atom in atoms:
        atom.tag = atom.index


@pytest.fixture
def bulk_al():
    return bulk("Al")


@pytest.fixture
def nacl():
    return bulk("NaCl", crystalstructure="rocksalt", a=4.0)


@pytest.fixture
def make_cluster_mng():
    def _make_cluster_mng(atoms, **kwargs):
        return ClusterManager(atoms, **kwargs)

    return _make_cluster_mng


@pytest.fixture
def cluster_mng(bulk_al, make_cluster_mng):
    return make_cluster_mng(bulk_al)


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
    prim = bulk("Al")
    prim[0].tag = 0
    manager = ClusterManager(prim)
    template = prim * (3, 3, 3)
    template.wrap()
    manager.build(max_cluster_dia=[3.0])
    trans_mat = manager.translation_matrix(template)
    assert trans_matrix_matches(trans_mat, template)


def test_tm_hcp():
    prim = bulk("Mg", crystalstructure="hcp", a=3.8, c=4.8)
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
    prim = bulk("LiX", crystalstructure="rocksalt", a=4.0)
    prim.wrap()
    for atom in prim:
        atom.tag = atom.index

    template = prim * (2, 2, 2)
    manager = ClusterManager(prim)
    manager.build(max_cluster_dia=[3.0])
    tm = manager.translation_matrix(template)
    assert trans_matrix_matches(tm, template)


def test_lut():
    prim = bulk("LiO", crystalstructure="rocksalt", a=4.05)
    prim.wrap()
    for atom in prim:
        atom.tag = atom.index

    tests = [
        {
            "atoms": prim,
            "expect": {FourVector(0, 0, 0, 0): 0, FourVector(0, 0, 0, 1): 1},
        },
        {
            "atoms": wrap_and_sort_by_position(prim * (2, 2, 2)),
            "expect": {
                FourVector(0, 0, 0, 0): 0,
                FourVector(0, 0, 0, 1): 4,
                FourVector(0, 1, 0, 0): 2,
                FourVector(0, 1, 0, 1): 9,
                FourVector(0, 0, 1, 0): 3,
                FourVector(0, 0, 1, 1): 10,
                FourVector(0, 1, 1, 0): 8,
                FourVector(0, 1, 1, 1): 14,
                FourVector(1, 0, 0, 0): 1,
                FourVector(1, 0, 0, 1): 7,
                FourVector(1, 0, 1, 0): 6,
                FourVector(1, 0, 1, 1): 13,
                FourVector(1, 1, 0, 0): 5,
                FourVector(1, 1, 0, 1): 12,
                FourVector(1, 1, 1, 0): 11,
                FourVector(1, 1, 1, 1): 15,
            },
        },
    ]

    manager = ClusterManager(prim)
    for i, test in enumerate(tests):
        lut = manager.create_four_vector_lut(test["atoms"])
        msg = "Test #{} failed.\nGot: {}\nExpected: {}".format(i, lut, test["expect"])
        assert lut == test["expect"], msg


def test_get_figures_multiple_times(cluster_mng):
    """Regression test. After getting figures from settings,
    attaching a calculator would result in a RuntimeError.

    See issue #263
    """
    fig1 = cluster_mng.get_figures()
    fig2 = cluster_mng.get_figures()
    assert fig1 == fig2


@pytest.mark.parametrize(
    "max_cluster_dia",
    [
        [5.0, 5.0],
        (5.0, 5.0),
        [],
        np.array([1, 2, 3]),
    ],
)
def test_cache(mocker, cluster_mng, max_cluster_dia):
    # max_size = 3
    # max_cluster_dia = [5, 5]
    mocker.spy(ClusterManager, "_prepare_new_build")
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
    max_cluster_dia = list(max_cluster_dia) + [5.0]
    assert cluster_mng.requires_build(max_cluster_dia)
    cluster_mng.build(max_cluster_dia)
    assert cluster_mng._prepare_new_build.call_count == 2
    # The first cluster must be the same object in memory, since
    # we didn't do anything to the cluster list
    assert cluster_0 is not cluster_mng.clusters[0]


@pytest.mark.parametrize(
    "background_syms, expect",
    [
        (None, {"Na", "Cl"}),
        ({"Na"}, {"Cl"}),
        ({"Cl"}, {"Na"}),
    ],
)
def test_background_manager_background(nacl, make_cluster_mng, background_syms, expect):
    """Test that we properly filter the background symbols in the primitive."""
    mng = make_cluster_mng(nacl, background_syms=background_syms)
    assert set(mng.prim.symbols) == expect


def test_no_mutation(nacl, make_cluster_mng):
    """Test we don't mutate the input atoms"""
    atoms = nacl
    atoms_orig = atoms.copy()
    mng = make_cluster_mng(atoms=atoms, background_syms={"Na"})
    # Test we filtered the primtive
    assert len(mng.prim) == 1
    assert set(mng.prim.symbols) == {"Cl"}
    # Test we didn't alter the atoms we inserted in the cluster manager
    assert atoms == atoms_orig
    assert set(atoms.symbols) == {"Na", "Cl"}


def test_is_background(nacl, make_cluster_mng):
    mng = make_cluster_mng(nacl, background_syms={"Na"})

    atoms = nacl * (4, 4, 4)
    assert set(atoms.symbols) == {"Na", "Cl"}
    for atom in atoms:
        expect = atom.symbol == "Na"
        assert mng.is_background_atom(atom) == expect


@pytest.mark.parametrize(
    "max_cluster_dia, max_body_size",
    [
        ([], 1),
        ([4.0], 2),
        ([4.0, 4.0], 3),
    ],
)
def test_build_empty(nacl, make_cluster_mng, max_cluster_dia, max_body_size):
    mng = make_cluster_mng(nacl)

    mng.build(max_cluster_dia=max_cluster_dia)

    counts = Counter(cluster.size for cluster in mng.clusters)
    print(counts)

    # We should have 1 empty and 2 singlets (2 sublattices)
    assert counts[0] == 1
    assert counts[1] == 2
    # Check we find the maximum size is the expected
    assert max(counts.keys()) == max_body_size


@pytest.mark.parametrize(
    "prim",
    [
        bulk("LiX", crystalstructure="rocksalt", a=4.0),
        bulk("Au", crystalstructure="fcc", a=3.8),
    ],
)
@pytest.mark.parametrize(
    "rep",
    [
        (1, 1, 1),
        (3, 3, 3),
        (1, 2, 3),
    ],
)
def test_trivial_path(prim, rep, make_cluster_mng, mocker):
    prim.wrap()
    tagger(prim)

    manager: ClusterManager = make_cluster_mng(prim)

    template = prim * rep

    # Ensure we're calling the correct wrapping function
    # We do not include background, so we assume the number of calls
    # is the number of atoms in the template
    mocker.spy(manager, "_wrap_four_vectors_trivial")
    mocker.spy(manager, "_wrap_four_vectors_general")

    manager.build(max_cluster_dia=[4.0, 4.0])
    assert manager._wrap_four_vectors_trivial.call_count == 0
    assert manager._wrap_four_vectors_general.call_count == 0

    tm_trivial = manager.translation_matrix(template)
    assert manager._wrap_four_vectors_trivial.call_count == len(template)
    assert manager._wrap_four_vectors_general.call_count == 0

    manager._allow_trivial_path = False
    tm_general = manager.translation_matrix(template)
    assert manager._wrap_four_vectors_trivial.call_count == len(template)
    assert manager._wrap_four_vectors_general.call_count == len(template)

    assert tm_trivial is not tm_general
    assert tm_trivial == tm_general


@pytest.mark.parametrize(
    "prim",
    [
        bulk("LiX", crystalstructure="rocksalt", a=4.0),
        bulk("Au", crystalstructure="fcc", a=3.8),
    ],
)
@pytest.mark.parametrize("max_cutoff_dia", [[5.0], [6.0, 5.0], [5.5, 5.5, 5.5]])
def test_cluster_list(prim, max_cutoff_dia, make_cluster_mng):
    prim.wrap()

    manager: ClusterManager = make_cluster_mng(prim)

    manager.build(max_cutoff_dia)

    for cluster in manager.clusters:
        exp_size = get_size_from_cf_name(cluster.name)
        assert cluster.size == exp_size
        for figure in cluster.figures:
            for fv in figure.components:
                fv._validate()
            max_size = max_cutoff_dia[figure.size - 2]
            dia = figure.get_diameter(prim)
            assert dia <= max_size
