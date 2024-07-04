import pytest
from ase.calculators.emt import EMT
from cleases.montecarlo.observers import (
    CorrelationFunctionObserver,
    EntropyProductionRate,
)
from cleases.montecarlo import KineticMonteCarlo, BEPBarrier, NeighbourSwap
from cleases.settings import CEBulk, Concentration
from cleases.calculator import attach_calculator
from cleases.montecarlo.mc_evaluator import MCEvaluator
from cleases.datastructures import MCStep


@pytest.fixture
def barrier():
    dilute_barriers = {"Au": 0.5, "Cu": 0.4}
    return BEPBarrier(dilute_barriers)


@pytest.fixture
def example_system(db_name):
    conc = Concentration(basis_elements=[["Au", "Cu", "X"]])
    settings = CEBulk(
        conc,
        crystalstructure="fcc",
        size=[1, 1, 1],
        max_cluster_dia=[3.0],
        db_name=db_name,
    )
    return settings


@pytest.fixture
def atoms(example_system):
    ats = example_system.atoms.copy() * (2, 2, 2)
    # Insert some Cu
    ats.symbols[:4] = "Cu"
    return ats


@pytest.fixture
def make_kmc(atoms, example_system, barrier):
    def _make_kmc(temp=300):
        settings = example_system
        eci = {"c0": 0.0, "c1_0": 0.0, "c2_d0000_0_00": 0.0}

        atoms_calc = attach_calculator(settings, atoms, eci)
        vac_idx = 5
        atoms_calc[vac_idx].symbol = "X"

        neighbor = NeighbourSwap(atoms_calc, 3.0)
        for l in neighbor.nl:
            assert len(l) == 12

        kmc = KineticMonteCarlo(atoms_calc, temp, barrier, [neighbor])
        return vac_idx, kmc

    return _make_kmc


def test_kmc(make_kmc, make_tempfile):
    vac_idx, kmc = make_kmc()
    atoms = kmc.atoms

    obs = CorrelationFunctionObserver(atoms.calc)
    kmc.attach(obs, 2)

    epr_file = make_tempfile("epr.txt")
    kmc.epr = EntropyProductionRate(buffer_length=2, logfile=epr_file)

    # Check that ValueError is raised if vac_idx is not vacancy
    with pytest.raises(ValueError):
        kmc.run(10, vac_idx - 1)
    kmc.run(10, vac_idx)

    # Just run reset to confirm that this method runs without error
    kmc.reset()


def test_kmc_step(make_kmc):
    vac_idx, kmc = make_kmc()
    for i in range(10):
        vac_idx, mc_step = kmc._mc_step(vac_idx, i)

        assert isinstance(mc_step, MCStep)
        assert mc_step.step == i
        assert isinstance(mc_step.other, dict)
        assert mc_step.other["time"] == kmc.time
        assert mc_step.move_accepted
        assert isinstance(mc_step.energy, float)


def test_kmc_emt(atoms, barrier):
    """Perform a KMC using an EMT calculator"""

    class EMTEvaluator(MCEvaluator):
        def get_energy(self, **kwargs):
            """Helper function to evaluate EMT energy with vacancies"""
            atoms = self.atoms
            mask = [atom.index for atom in atoms if atom.symbol != "X"]
            atoms_masked = atoms[mask]
            atoms_masked.calc = EMT()
            return atoms_masked.get_potential_energy()

    evaluator = EMTEvaluator(atoms)

    vac_idx = 5
    atoms[vac_idx].symbol = "X"
    neighbor = NeighbourSwap(atoms, 3.0)
    orig_symbols = list(atoms.symbols)

    kmc = KineticMonteCarlo(evaluator, 300, barrier, [neighbor])

    # Check we can evaluate rates
    swaps, rates = kmc._rates(vac_idx)
    # Verify the rates are non-zero, i.e. they seem sensible
    assert len(swaps) > 0
    assert all(rates > 0)

    assert list(atoms.symbols) == orig_symbols
    # Verify we can run the KMC
    kmc.run(5, vac_idx)
    assert list(atoms.symbols) != orig_symbols
