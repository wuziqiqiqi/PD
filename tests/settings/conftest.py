import pytest
from clease.settings import Concentration


@pytest.fixture
def make_conc():

    def _make_conc(basis_elements, **kwargs):
        return Concentration(basis_elements=basis_elements, **kwargs)

    return _make_conc
