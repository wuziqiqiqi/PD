import os
import pytest
from clease_cxx import has_parallel


@pytest.fixture(autouse=True)
def needs_parallel():
    """Tests in this test suite should not be able to run if
    not compiled with OpenMP."""
    assert has_parallel(), "CLEASE has not been configured with OpenMP threading."


@pytest.fixture
def cpu_count():
    return os.cpu_count()
