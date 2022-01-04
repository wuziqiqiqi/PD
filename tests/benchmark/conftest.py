import pytest


@pytest.fixture(autouse=True)
def require_benchmark(benchmark):
    """Any tests in this folder will be skipped if benchmarks are disabled/skipped."""
