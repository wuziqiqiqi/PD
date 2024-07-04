import random
import pytest
from clease.datastructures import FourVector


@pytest.fixture
def make_random_four_vector():
    def _make_random_four_vector(min=-3, max=3):
        ints = (random.randint(min, max) for _ in range(4))
        return FourVector(*ints)

    return _make_random_four_vector
