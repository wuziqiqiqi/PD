import random
import pytest


@pytest.fixture
def make_random_eci():
    """Construct a randomized ECI dictionary from a settings object."""

    def _make(settings, vmin=-2, vmax=2):
        eci = {name: random.uniform(vmin, vmax) for name in settings.all_cf_names}
        return eci

    return _make
