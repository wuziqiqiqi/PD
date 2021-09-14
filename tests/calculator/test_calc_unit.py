import pytest
import numpy as np
from clease.calculator import Clease
from clease.calculator import attach_calculator
from clease.settings import Concentration


@pytest.fixture
def make_atoms(get_LiVX, get_random_eci):
    """Factory fixture for making LiVX atoms, and attaching a calculator"""

    def _make_atoms(rep=(1, 1, 1), **kwargs):
        settings = get_LiVX(**kwargs)
        atoms = settings.atoms * rep
        eci = get_random_eci(settings)
        atoms = attach_calculator(settings, atoms, eci)
        return atoms

    return _make_atoms


@pytest.fixture
def make_dummy_atoms(make_dummy_settings, get_random_eci):

    def _make_dummy_atoms(rep=(1, 1, 1), **kwargs):
        settings = make_dummy_settings(**kwargs)
        atoms = settings.atoms * rep
        eci = get_random_eci(settings)
        atoms = attach_calculator(settings, atoms, eci)
        return atoms

    return _make_dummy_atoms


def test_get_energy(make_atoms):
    atoms = make_atoms(size=[1, 1, 1])
    en1 = atoms.get_potential_energy()
    assert isinstance(en1, float)


@pytest.mark.parametrize('invalid_settings',
                         ['somestring', None, True, False,
                          Concentration(basis_elements=[['Au']])])
def test_invalid_settings(invalid_settings):
    """Test that if we pass some things which are not settings,
    that we cannot initialize"""
    with pytest.raises(TypeError):
        Clease(invalid_settings, {})


def test_init_cf(make_dummy_settings, get_random_eci):
    settings = make_dummy_settings()
    eci = {}

    # This should be fine
    Clease(settings, eci)
    Clease(settings, eci, init_cf=None)
    Clease(settings, eci, init_cf={})

    # Wrong type of init_cf
    with pytest.raises(TypeError):
        Clease(settings, eci, init_cf=True)
    with pytest.raises(TypeError):
        Clease(settings, eci, init_cf=['a', 'b', 'c'])

    # Fill ECI with some values, so it's not empty
    # ECI and init_cf need to match in length
    eci = get_random_eci(settings)
    assert len(eci) > 0
    with pytest.raises(ValueError):
        Clease(settings, eci, init_cf={})
    with pytest.raises(ValueError):
        Clease(settings, {}, init_cf={'a': 1})

    # Just use ECI as the cf dict - same length
    Clease(settings, eci, init_cf=eci)


def test_set_atoms(make_dummy_atoms):
    """Test changing the internal atoms in the calculator"""

    def _initialize():
        atoms = make_dummy_atoms(rep=(2, 2, 2))
        calc = atoms.calc
        atoms2 = atoms.copy()
        assert atoms2 == calc.atoms
        assert atoms2 is not calc.atoms  # Must be different in memory

        return atoms2, calc

    # Set the same atoms again
    atoms, calc = _initialize()
    calc.set_atoms(atoms)

    # Use a single symbol, should be fine
    atoms, calc = _initialize()
    atoms.symbols[:] = atoms.symbols[0]
    calc.set_atoms(atoms)

    # Set with randomized symbols
    atoms, calc = _initialize()
    unique_symbols = sorted(set(atoms.symbols))
    for _ in range(10):
        new_symbols = np.random.choice(unique_symbols, size=len(atoms))
        atoms.symbols = new_symbols
        calc.set_atoms(atoms)

    # Wrong number of atoms
    atoms, calc = _initialize()
    del atoms[0]
    with pytest.raises(ValueError):
        calc.set_atoms(atoms)

    # Changed positions
    atoms, calc = _initialize()
    atoms[0].x += 0.01
    with pytest.raises(ValueError):
        calc.set_atoms(atoms)



def test_only_empty_and_singlet(make_dummy_settings):
    # Test added due to issue #265: seg fault
    settings = make_dummy_settings(max_cluster_size=1, max_cluster_dia=())
    eci = {'c0': 0.0, 'c1_0': 0.0}
    calc = Clease(settings, eci)
    atoms = settings.atoms.copy()

    # This line produced segfault
    atoms.calc = calc

    # Add some assertions that confirms lengths of the cluster list
    assert len(settings.cluster_list) == 2
    
    # Make sure that all cluster sizes are 0 or one
    for cluster in settings.cluster_list:
        assert cluster.size in (0, 1)
