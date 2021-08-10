import os
import pytest
from ase.calculators.emt import EMT
from clease.montecarlo.observers import CorrelationFunctionObserver, EntropyProductionRate
from clease.montecarlo import KineticMonteCarlo, BEPBarrier, NeighbourSwap
from clease.settings import CEBulk, Concentration
from clease.calculator import attach_calculator
from clease.montecarlo.mc_evaluator import MCEvaluator


@pytest.fixture
def rng(make_rng):
    return make_rng(44)


@pytest.fixture
def barrier():
    dilute_barriers = {'Au': 0.5, 'Cu': 0.4}
    return BEPBarrier(dilute_barriers)


@pytest.fixture
def example_system(db_name):
    conc = Concentration(basis_elements=[['Au', 'Cu', 'X']])
    settings = CEBulk(conc,
                      crystalstructure='fcc',
                      size=[1, 1, 1],
                      max_cluster_size=2,
                      max_cluster_dia=[3.0],
                      db_name=db_name)
    return settings


@pytest.fixture
def atoms(example_system):
    ats = example_system.atoms.copy() * (2, 2, 2)
    # Insert some Cu
    ats.symbols[:4] = 'Cu'
    return ats


def test_kmc(atoms, example_system, barrier, rng):
    settings = example_system
    eci = {'c0': 0.0, 'c1_0': 0.0, 'c2_d0000_0_00': 0.0}

    atoms = attach_calculator(settings, atoms, eci)
    vac_idx = 5
    atoms[vac_idx].symbol = 'X'

    neighbor = NeighbourSwap(atoms, 3.0)
    for l in neighbor.nl:
        assert len(l) == 12

    T = 300
    kmc = KineticMonteCarlo(atoms, T, barrier, [neighbor], rng=rng)
    obs = CorrelationFunctionObserver(atoms.calc)
    kmc.attach(obs, 2)

    epr_file = 'epr.txt'
    kmc.epr = EntropyProductionRate(buffer_length=2, logfile=epr_file)

    # Check that ValueError is raised if vac_idx is not vacancy
    with pytest.raises(ValueError):
        kmc.run(10, vac_idx - 1)
    kmc.run(10, vac_idx)

    # Just run reset to confirm that this method runs without error
    kmc.reset()
    os.remove(epr_file)


def test_kmc_emt(atoms, rng, barrier):
    """Perform a KMC using an EMT calculator"""

    class EMTEvaluator(MCEvaluator):

        def get_energy(self, **kwargs):
            """Helper function to evaluate EMT energy with vacancies"""
            atoms = self.atoms
            mask = [atom.index for atom in atoms if atom.symbol != 'X']
            atoms_masked = atoms[mask]
            atoms_masked.calc = EMT()
            return atoms_masked.get_potential_energy()

    evaluator = EMTEvaluator(atoms)

    vac_idx = 5
    atoms[vac_idx].symbol = 'X'
    neighbor = NeighbourSwap(atoms, 3.0)
    orig_symbols = list(atoms.symbols)

    kmc = KineticMonteCarlo(evaluator, 300, barrier, [neighbor], rng=rng)

    # Check we can evaluate rates
    swaps, rates = kmc._rates(vac_idx)
    # Verify the rates are non-zero, i.e. they seem sensible
    assert len(swaps) > 0
    assert all(rates > 0)

    assert list(atoms.symbols) == orig_symbols
    # Verify we can run the KMC
    kmc.run(5, vac_idx)
    assert list(atoms.symbols) != orig_symbols
