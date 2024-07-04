import pytest
from clease.cluster import Cluster, ClusterFingerprint


@pytest.fixture
def make_dummy_cluster():
    def _make_dummy_cluster(
        name="c0",
        size=0,
        diameter=0.0,
        fingerprint=ClusterFingerprint((0.0,)),
        figures=(),
        equiv_sites=(),
        group=0,
    ):
        return Cluster(
            name=name,
            size=size,
            diameter=diameter,
            fingerprint=fingerprint,
            figures=figures,
            equiv_sites=equiv_sites,
            group=group,
        )

    return _make_dummy_cluster


@pytest.mark.parametrize(
    "test",
    [
        {
            "deco": [1, 2, 3, 4],
            "equiv_site": [[0, 1, 2]],
            "result": [
                [1, 2, 3, 4],
                [1, 3, 2, 4],
                [2, 1, 3, 4],
                [2, 3, 1, 4],
                [3, 1, 2, 4],
                [3, 2, 1, 4],
            ],
        },
        {
            "deco": [1, 2, 3, 4],
            "equiv_site": [[0, 3]],
            "result": [[1, 2, 3, 4], [4, 2, 3, 1]],
        },
        {"deco": [1, 2, 3, 4], "equiv_site": [], "result": [[1, 2, 3, 4]]},
    ],
)
def test_equiv_deco(make_dummy_cluster, test):
    cluster = make_dummy_cluster(equiv_sites=test["equiv_site"])
    assert cluster.equiv_deco(test["deco"]) == test["result"]
