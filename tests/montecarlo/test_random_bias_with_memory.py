import os
import pytest
from ase.build import bulk
from clease.montecarlo import RandomBiasWithMemory
from clease.settings import CEBulk, Concentration
from clease.montecarlo import Montecarlo
from clease.calculator import attach_calculator


@pytest.fixture
def atoms_ref():
    return bulk('Fe') * (4, 4, 4)


def test_hash_key_consistency(atoms_ref):
    bias = RandomBiasWithMemory(1.0, atoms_ref)
    all_atoms = []

    hash_keys = []
    for i in range(40):
        atoms_ref[i].symbol = 'C'
        hash_keys.append(bias.get_hash_key())
        all_atoms.append(atoms_ref.copy())

    # Check that the keys are consistent
    for i in range(40):
        atoms_ref.numbers = all_atoms[i].numbers
        key = bias.get_hash_key()
        assert key == hash_keys[i]


def test_history_consistency(atoms_ref):
    orig = atoms_ref.copy()
    swaps = [{'idx': i, 'symb': 'C'} for i in range(30)]
    bias = RandomBiasWithMemory(1.0, atoms_ref)

    hash_keys = []
    for i, s in enumerate(swaps):
        atoms_ref[s['idx']].symbol = s['symb']
        bias(None)
        hash_keys.append(bias.get_hash_key())
        assert len(bias.history) == i + 1

    atoms_ref.numbers = orig.numbers

    for i, s in enumerate(swaps):
        atoms_ref[s['idx']].symbol = s['symb']
        corr = bias(None)
        assert corr == pytest.approx(bias.history[hash_keys[i]])


def test_attached_to_mc(db_name):
    conc = Concentration(basis_elements=[['Al', 'Mg']])
    settings = CEBulk(conc,
                      db_name=db_name,
                      a=4.05,
                      max_cluster_dia=[3.0],
                      crystalstructure='fcc',
                      size=[4, 4, 4])

    atoms = settings.atoms.copy()
    atoms = attach_calculator(settings=settings, atoms=atoms, eci={'c0': 0.0, 'c2_d0000_0_00': 1.0})

    atoms[0].symbol = 'Mg'
    atoms[1].symbol = 'Mg'
    mc = Montecarlo(atoms, 10000)

    bias = RandomBiasWithMemory(0.5, atoms)
    mc.add_bias(bias)

    mc.run(steps=10)
