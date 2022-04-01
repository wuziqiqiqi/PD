import pytest
from pathlib import Path
from clease.settings import ClusterExpansionSettings


@pytest.fixture
def references_path():
    """Path to the references directory."""
    return Path(__file__).parent / "references"


@pytest.fixture
def verify_clusters():
    """Helper fixture to do some tests on the clusters in the settings object."""

    def _verify_clusters(settings: ClusterExpansionSettings):
        mcd = settings.max_cluster_dia
        prim = settings.prim_cell
        for cluster in settings.cluster_list:
            for figure in cluster.figures:
                for fv in figure.components:
                    fv._validate()
                max_size = mcd[figure.size - 2]
                dia = figure.get_diameter(prim)
                assert dia <= max_size

    return _verify_clusters
