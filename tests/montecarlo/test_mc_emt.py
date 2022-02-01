"""Test the Monte-Carlo algorithm without a Cluster Expansion calculator, but instead
using an EMT calculator."""
import random
import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
import clease
from clease.datastructures import SystemChange
from clease.montecarlo import Montecarlo, MCEvaluator


@pytest.fixture
def calc():
    return EMT()


@pytest.fixture
def atoms(calc) -> Atoms:
    ats = bulk("Cu") * (4, 4, 4)
    assert len(ats) == 64
    ats.symbols[:30] = "Au"

    # Randomize the symbols
    syms = list(ats.symbols)
    random.shuffle(syms)
    ats.symbols = syms
    ats.calc = calc
    return ats


@pytest.fixture
def evaluator(atoms):
    return MCEvaluator(atoms)


def test_emt_mc_implicit_evaluator(atoms):
    """Test running MC using EMT, implicitly constructing the MC evaluator"""
    E0 = atoms.get_potential_energy()
    for temp in [1000, 500, 100]:
        mc = Montecarlo(atoms, temp)
        mc.run(steps=5)
    # Check we created the correct type of evaluator
    assert isinstance(mc.evaluator, clease.montecarlo.mc_evaluator.MCEvaluator)
    assert not isinstance(mc.evaluator, clease.montecarlo.mc_evaluator.CEMCEvaluator)
    E1 = atoms.get_potential_energy()

    # Check the energy has changed
    assert abs(E0 - E1) > 1e-4
    assert E1 < E0


def test_emt_mc_explicit_evaluator(atoms, evaluator):
    """Test running MC using EMT, passing in an explicitly constructed MC evaluator"""
    # Set the seed for the MC algorithm - does not yet support receiving an RNG object.
    E0 = atoms.get_potential_energy()
    assert E0 == pytest.approx(evaluator.get_energy())
    for temp in [1000, 500, 100]:
        mc = Montecarlo(evaluator, temp)
        # Ensure we use the exact instance of the evaluator
        assert mc.evaluator is evaluator
        assert mc.atoms is evaluator.atoms
        mc.run(steps=5)

    E1 = atoms.get_potential_energy()

    # Check the energy has changed
    assert abs(E0 - E1) > 1e-4
    assert E1 < E0


def test_evaluator(evaluator):
    atoms = evaluator.atoms
    en = evaluator.get_energy()
    assert isinstance(en, float)

    original_symbols = list(atoms.symbols)
    # Change the first symbol to something not already in the list of symbols
    idx = 0
    old_symb = original_symbols[idx]
    new_symb = "Ni"
    assert old_symb != new_symb
    assert new_symb not in set(original_symbols)
    changes = [SystemChange(idx, old_symb, new_symb, "test")]

    evaluator.apply_system_changes(changes)
    assert atoms.symbols[0] == new_symb
    en1 = evaluator.get_energy()
    assert abs(en - en1) > 1e-4

    evaluator.undo_system_changes(changes)
    assert list(atoms.symbols) == original_symbols
    assert pytest.approx(evaluator.get_energy(), en)
