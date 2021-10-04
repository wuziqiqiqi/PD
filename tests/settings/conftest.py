import pytest
from pathlib import Path


@pytest.fixture
def references_path():
    """Path to the references directory."""
    return Path(__file__).parent / 'references'
