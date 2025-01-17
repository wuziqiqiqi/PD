import os
from pathlib import Path
import json
import random
from collections import defaultdict
import pytest
import numpy as np
from ase import Atoms
from ase.units import kB
from ase.build import bulk
from ase.geometry import get_layers
import clease
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
from clease.montecarlo.observers import CorrelationFunctionObserver
from clease.montecarlo.observers import Snapshot
from clease.montecarlo.observers import EnergyEvolution
from clease.montecarlo.observers import SiteOrderParameter
from clease.montecarlo.observers import LowestEnergyStructure
from clease.montecarlo.observers import DiffractionObserver, AcceptanceRate, MCObserver
from clease.montecarlo.constraints import ConstrainSwapByBasis, FixedElement
from clease.montecarlo import RandomSwap, MixedSwapFlip
from clease.settings import CEBulk, Concentration
from clease.corr_func import CorrFunction
from clease.datastructures import SystemChange, MCStep
from clease.montecarlo.mc_evaluator import CEMCEvaluator

# Set the random seed
np.random.seed(0)

almgsix_eci_file = Path(__file__).parent / "almgsix_eci.json"


class DummyObs(MCObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def observe_step(self, mc_step: MCStep) -> None:
        self.call_count += 1


@pytest.fixture
def almgsix_eci():
    with almgsix_eci_file.open() as file:
        return json.load(file)


@pytest.fixture
def make_settings_with_bkg(db_name):
    def _make_settings_with_bkg(size=(1, 1, 1)):
        conc = Concentration(basis_elements=[["Li", "X"], ["O"]])
        settings = CEBulk(
            db_name=db_name,
            concentration=conc,
            a=4.0,
            crystalstructure="rocksalt",
            max_cluster_dia=[5.0],
            size=size,
        )
        atoms = settings.atoms.copy()
        non_bkg_sites = atoms.symbols == "Li"
        # Set half of the symbols to "X"
        # (randomly ordered)
        indices = list(np.arange(len(atoms))[non_bkg_sites])
        indices = random.sample(indices, len(indices) // 2)
        atoms.symbols[indices] = "X"

        all_cf_names = settings.all_cf_names
        eci = {name: random.random() for name in all_cf_names}
        atoms = attach_calculator(settings, atoms, eci=eci)

        return settings, atoms

    return _make_settings_with_bkg


def get_example_mc_system(db_name):
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    settings = CEBulk(
        db_name=db_name,
        concentration=conc,
        crystalstructure="fcc",
        a=4.0,
        max_cluster_dia=[5.0, 4.1],
        size=[2, 2, 2],
    )

    atoms = settings.atoms.copy() * (3, 3, 3)
    # Insert a few different symbols
    atoms.symbols = "Au"
    atoms.symbols[:10] = "Cu"
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}
    eci["c0"] = 1.0
    eci["c2_d0000_0_00"] = 2.5
    eci["c3_d0000_0_000"] = 3.5
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    return atoms


def get_rocksalt_mc_system(db_name):
    conc = Concentration(basis_elements=[["Si", "X"], ["O", "C"]])
    settings = CEBulk(
        db_name=db_name,
        concentration=conc,
        crystalstructure="rocksalt",
        a=4.0,
        max_cluster_dia=[2.51, 3.0],
        size=[2, 2, 2],
    )
    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}
    eci["c0"] = -1.0
    eci["c2_d0000_0_00"] = -2.5
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    return atoms


@pytest.fixture
def example_system(db_name):
    return get_rocksalt_mc_system(db_name)


@pytest.mark.slow
def test_run_heavy(db_name):
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    settings = CEBulk(
        db_name=db_name,
        concentration=conc,
        crystalstructure="fcc",
        a=4.0,
        max_cluster_dia=[5, 5, 5],
    )

    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci["c0"] = -1.0
    eci["c2_d0000_0_00"] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    # Insert a few elements
    for i in range(10):
        atoms[i].symbol = "Cu"

    E = []
    for T in [1000, 500, 100]:
        mc = Montecarlo(atoms, T)
        mc.run(steps=10_000)
        E.append(mc.get_thermodynamic_quantities()["energy"])

    cf = CorrFunction(atoms.calc.settings)
    cf_calc = atoms.calc.get_cf()
    cf_scratch = cf.get_cf(atoms)

    print(len(cf_scratch))

    for k, v in cf_scratch.items():
        assert v == pytest.approx(cf_calc[k])

    # Make sure that the energies are decreasing by comparing
    # the first and last
    assert E[0] >= E[-1]


def test_run(db_name):
    atoms = get_example_mc_system(db_name)

    E = []
    mc = Montecarlo(atoms, 300)
    for T in [1000, 500, 100]:
        mc.temperature = T
        # Verify we don't divide by 0, and that steps are reset.
        assert mc.current_accept_rate == 0
        mc.run(steps=1000)
        assert 0 <= mc.current_accept_rate <= 1
        thermo = mc.get_thermodynamic_quantities()
        assert "accept_rate" in thermo
        E.append(thermo["energy"])

    assert isinstance(mc.evaluator, clease.montecarlo.mc_evaluator.CEMCEvaluator)
    assert mc.atoms is atoms

    cf = CorrFunction(atoms.calc.settings)
    cf_calc = atoms.calc.get_cf()
    cf_scratch = cf.get_cf(atoms)

    assert cf_calc.keys() == cf_scratch.keys()

    for k, v in cf_scratch.items():
        assert v == pytest.approx(cf_calc[k]), k
    # Make sure that the energies are decreasing by comparing
    # the first and last
    assert E[0] >= E[-1]


def test_mc_rng(db_name, set_rng):
    """Test passing in an explicit rng object with same seeds
    produce identical MC runs.

    First two runs have same seed, final run is a different seed
    """
    atoms = get_example_mc_system(db_name)

    # Grab a copy of the initial configuration
    ini_syms = list(atoms.symbols)

    energies = defaultdict(list)
    state_before = defaultdict(list)
    state_after = defaultdict(list)

    seeds = [8, 9, 9, 10, 42, 10, 42, 8, 8, 42, 10, 10]

    start_energies = []
    for seed in seeds:
        set_rng(seed)
        # Get the state of random before we start
        state_before[seed].append(random.getstate())

        # Reset the symbols, as they are mutated during an MC run
        atoms.symbols = ini_syms
        # Resync the calculator
        start_energies.append(atoms.get_potential_energy())

        mc = Montecarlo(atoms, 200_000)
        mc.run(steps=500)
        # Get the state after we're done, so we can compare
        state_after[seed].append(random.getstate())
        energies[seed].append(mc.get_thermodynamic_quantities()["energy"])

    # Test the energies after resetting are identical
    assert np.array(start_energies) == pytest.approx(start_energies[0])

    # Check that the random states before and after MC are identical
    # when we start from the same seed
    for ii, state_dict in enumerate((state_before, state_after)):
        for seed in seeds:
            values = state_dict[seed]
            state0 = values[0]
            assert all(state == state0 for state in values), (ii, seed)

    for seed, en in energies.items():
        # Check all values are identical
        assert np.array(en) == pytest.approx(en[0]), energies
        # Check that all values starting with other seeds are sufficiently different
        for seed2, en2 in energies.items():
            if seed == seed2:
                continue
            # We already checked each entry in the energies are equivalent,
            # so just check the first elements are different between different seeds
            assert abs(en[0] - en2[0]) > 1e-3


def test_corr_func_observer(db_name):
    atoms = get_example_mc_system(db_name)

    atoms[0].symbol = "Cu"
    atoms[1].symbol = "Cu"

    mc = Montecarlo(atoms, 600)
    obs = CorrelationFunctionObserver(atoms.calc)
    mc.attach(obs, interval=1)
    mc.run(steps=500)
    thermo = mc.get_thermodynamic_quantities()
    _ = obs.get_averages()
    assert obs.counter == 501

    cf_keys = atoms.calc.get_cf().keys()

    for k in cf_keys:
        assert "cf_" + k in thermo.keys()


def test_snapshot(db_name, make_tempfile):
    atoms = get_example_mc_system(db_name)

    fname = make_tempfile("snapshot.traj")
    obs = Snapshot(atoms, fname=fname)

    mc = Montecarlo(atoms, 600)
    mc.attach(obs, interval=100)
    mc.run(steps=1000)
    assert len(obs.traj) == 10

    # Test extension-less
    fname = make_tempfile("snapshot")
    obs = Snapshot(atoms, fname=fname)
    assert str(obs.fname) == fname + ".traj"


def test_energy_evolution(db_name, make_tempfile):
    atoms = get_example_mc_system(db_name)

    mc = Montecarlo(atoms, 600)
    obs = EnergyEvolution(mc)
    mc.attach(obs, interval=50)
    mc.run(steps=1000)

    # Just confirm that the save function works
    fname = make_tempfile("energy_evol.csv")
    obs.save(fname=fname)
    assert os.path.isfile(fname)

    # Check the number of energy values
    assert len(obs.energies) == 20

    # Test extensionless, should default to .csv
    fname_base = "energy_evol_no_ext"
    fname = make_tempfile(fname_base)
    obs.save(fname=fname)
    assert os.path.isfile(str(fname) + ".csv")
    assert not os.path.isfile(fname)


def test_site_order_parameter(db_name):
    atoms = get_example_mc_system(db_name)

    # Manually change some symbols
    atoms.symbols = "Au"
    atoms.symbols[:3] = "Cu"

    obs = SiteOrderParameter(atoms)
    mc = Montecarlo(atoms, 600)
    mc.attach(obs)
    mc.run(steps=1000)
    avg = obs.get_averages()
    assert avg["site_order_average"] <= 6.0

    thermo = mc.get_thermodynamic_quantities()

    assert "site_order_average" in thermo.keys()
    assert "site_order_std" in thermo.keys()


def test_lowest_energy_obs(db_name, compare_atoms):
    atoms = get_example_mc_system(db_name)
    atoms.symbols[0:3] = "Cu"

    assert len(set(atoms.symbols)) > 1

    low_en = LowestEnergyStructure(atoms)

    assert low_en.emin_atoms is not atoms
    assert isinstance(low_en.emin_atoms, Atoms)

    mc = Montecarlo(atoms, 700)
    energy_evol = EnergyEvolution(mc)
    mc.attach(low_en, interval=1)
    mc.attach(energy_evol, interval=1)

    mc.run(steps=1000)
    assert np.min(energy_evol.energies) == pytest.approx(low_en.lowest_energy)
    assert low_en.emin_atoms.get_potential_energy() == pytest.approx(low_en.lowest_energy)

    # Verify that mutating the original atoms object doesn't mutate the emin atoms object.
    nums = np.array(low_en.emin_atoms.numbers)  # Copy the numbers
    atoms.numbers[0] = 88
    assert (low_en.emin_atoms.numbers == nums).all()
    # Verify we can get the energy property, without ASE failing the check_state test.
    low_en.emin_atoms.get_potential_energy()


def test_track_lowest_cf(db_name):
    atoms = get_example_mc_system(db_name)
    atoms.symbols[0:3] = "Cu"

    low_en = LowestEnergyStructure(atoms)

    assert low_en.emin_atoms is not atoms
    assert low_en._calc_cache is low_en.emin_atoms.calc
    assert isinstance(low_en.emin_atoms, Atoms)
    mc = Montecarlo(atoms, 700)
    mc.attach(low_en)

    # Tracking CF's is disabled by default.
    assert low_en.lowest_energy_cf is None
    assert low_en.lowest_energy == np.inf
    mc.run(500)
    assert low_en.lowest_energy < np.inf
    assert low_en.lowest_energy == pytest.approx(low_en._lowest_energy_cache)
    assert low_en.lowest_energy_cf is None
    assert low_en._calc_cache is low_en.emin_atoms.calc
    assert low_en.emin_atoms.get_potential_energy() == pytest.approx(low_en.lowest_energy)

    # Enable tracking CF, should now get tracked.
    low_en.track_cf = True
    low_en.reset()
    assert low_en.lowest_energy == np.inf
    mc.run(500)
    assert low_en.lowest_energy_cf is not None
    assert isinstance(low_en.lowest_energy_cf, dict)
    for k, v in low_en.lowest_energy_cf.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
    assert low_en.lowest_energy == pytest.approx(low_en._lowest_energy_cache)


def test_diffraction_obs(db_name):
    atoms = get_example_mc_system(db_name)

    atoms[0].symbol = "Cu"
    atoms[1].symbol = "Cu"
    atoms[2].symbol = "Cu"

    obs = DiffractionObserver(
        atoms=atoms,
        k_vector=[0.25, 0.0, 0.0],
        name="reflect1",
        active_symbols=["Cu"],
        all_symbols=["Au", "Cu"],
    )

    mc = Montecarlo(atoms, 600)
    mc.attach(obs)

    mc.run(steps=1000)
    thermo = mc.get_thermodynamic_quantities()

    assert "reflect1" in thermo.keys()


def test_constrain_swap(db_name):
    atoms = get_rocksalt_mc_system(db_name)
    settings = atoms.calc.settings
    i_by_basis = settings.index_by_basis

    # Insert a few vacancies
    num_X = 0
    for atom in atoms:
        if atom.symbol == "Si":
            atom.symbol = "X"
            num_X += 1

        if num_X >= 20:
            break

    # Insert a few C
    num_C = 0
    for atom in atoms:
        if atom.symbol == "O":
            atom.symbol = "C"
            num_C += 1

        if num_C >= 20:
            break

    orig_symbs = [atom.symbol for atom in atoms]

    cnst = ConstrainSwapByBasis(atoms, i_by_basis)
    generator = RandomSwap(atoms)
    generator.add_constraint(cnst)

    mc = Montecarlo(atoms, 600, generator=generator)
    mc.run(steps=1000)

    # Confirm that swaps have been made
    symbs = [atom.symbol for atom in atoms]
    assert not all(map(lambda x: x[0] == x[1], zip(orig_symbs, symbs)))

    allowed_elements = [["Si", "X"], ["O", "C"]]
    for basis, allowed in zip(i_by_basis, allowed_elements):
        for indx in basis:
            assert atoms[indx].symbol in allowed


def test_fixed_element(db_name):
    atoms = get_rocksalt_mc_system(db_name)

    # Insert a few vacancies
    num_X = 0
    for atom in atoms:
        if atom.symbol == "Si":
            atom.symbol = "X"
            num_X += 1

        if num_X >= 20:
            break

    # Insert a few C
    num_C = 0
    for atom in atoms:
        if atom.symbol == "O":
            atom.symbol = "C"
            num_C += 1

        if num_C >= 20:
            break

    # Let's say that all Si atoms should be fixed
    fixed_element = FixedElement("Si")
    generator = RandomSwap(atoms)
    generator.add_constraint(fixed_element)

    mc = Montecarlo(atoms, 10000, generator=generator)

    si_indices = [atom.index for atom in atoms if atom.symbol == "Si"]
    mc.run(steps=1000)

    si_after = [atom.index for atom in atoms if atom.symbol == "Si"]
    assert si_indices == si_after


def test_fixed_indices(db_name):
    atoms = get_rocksalt_mc_system(db_name)

    # Insert a few vacancies
    num_X = 0
    for atom in atoms:
        if atom.symbol == "Si":
            atom.symbol = "X"
            num_X += 1

        if num_X >= 20:
            break

    # Insert a few C
    num_C = 0
    for atom in atoms:
        if atom.symbol == "O":
            atom.symbol = "C"
            num_C += 1

        if num_C >= 20:
            break

    # Fix the first n atoms
    n = 80
    indices = list(range(n, len(atoms)))
    generator = RandomSwap(atoms, indices)

    mc = Montecarlo(atoms, 10000, generator=generator)

    syms_before = atoms.symbols[:n]
    remainder_before = list(atoms.symbols[n:])
    mc.run(steps=2000)

    syms_after = atoms.symbols[:n]
    remainder_after = list(atoms.symbols[n:])
    assert all(syms_before == syms_after)
    # Check that we didn't constrain the other symbols
    assert remainder_before != remainder_after


@pytest.mark.slow
# pylint: disable=redefined-outer-name
def test_gs_mgsi(db_name, almgsix_eci):
    conc = Concentration(basis_elements=[["Al", "Mg", "Si", "X"]])
    settings = CEBulk(
        conc,
        crystalstructure="fcc",
        a=4.05,
        size=[1, 1, 1],
        max_cluster_dia=[5.0, 5.0],
        db_name=db_name,
    )
    settings.basis_func_type = "binary_linear"

    atoms = bulk("Al", a=4.05, cubic=True) * (2, 2, 2)
    atoms = attach_calculator(settings, atoms, almgsix_eci)
    expect = atoms.copy()
    layers, _ = get_layers(expect, (1, 0, 0))
    for i, layer in enumerate(layers):
        if layer % 2 == 0:
            expect[i].symbol = "Si"
        else:
            expect[i].symbol = "Mg"

    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Mg"
        atoms[i + int(len(atoms) / 2)].symbol = "Si"

    mc = Montecarlo(atoms, 1000)
    print(mc.current_energy)
    en_obs = EnergyEvolution(mc)
    mc.attach(en_obs)
    temps = [1000, 800, 600, 500, 400, 300, 200, 100]
    for T in temps:
        mc.temperature = T
        mc.run(steps=100 * len(atoms))

    E_final = atoms.get_potential_energy()
    atoms.numbers[:] = expect.numbers
    E_expect = atoms.get_potential_energy()

    cf = CorrFunction(settings)
    cf_final = cf.get_cf(atoms)
    cf_calc = atoms.calc.get_cf()

    for k in cf_calc.keys():
        assert cf_calc[k] == pytest.approx(cf_final[k], abs=1e-6)

    # Check that the expected energy is as it should be
    assert E_expect == pytest.approx(-108.67689884003414, abs=1e-6)

    # Due to simulated annealing and some round-off issues in acceptance criteria
    # the ground state may not be found within the specified amount of runs.
    # We check only that the energy is between E_expect +- 1
    assert E_final == pytest.approx(E_expect, abs=1.0)


def test_acceptance_rate(db_name):
    acc_rate = AcceptanceRate()
    assert acc_rate.rate == pytest.approx(0.0)

    num_acc = 3
    for _ in range(num_acc):
        acc_rate([SystemChange(0, "Al", "Mg", "")])

    assert acc_rate.rate == pytest.approx(1.0)

    num_reject = 3
    for _ in range(num_reject):
        acc_rate([SystemChange(0, "Al", "Al", "")])
    assert acc_rate.rate == pytest.approx(num_acc / (num_acc + num_reject))

    # Try to use as observer in MC
    atoms = get_example_mc_system(db_name)

    # Use very high temp to make sure that moves are accepted
    mc = Montecarlo(atoms, 50000)
    acc_rate.reset()
    assert acc_rate.num_calls == 0
    assert acc_rate.num_accept == 0
    assert acc_rate.rate == pytest.approx(0.0)

    mc.attach(acc_rate)
    mc.run(steps=10)
    assert acc_rate.num_accept > 0
    assert acc_rate.num_calls > 0


def test_mc_mixed_ensemble(db_name):
    atoms = get_rocksalt_mc_system(db_name)
    fixed_conc = [atom.index for atom in atoms if atom.symbol == "Si"]
    fixed_chem_pot = [atom.index for atom in atoms if atom.symbol == "O"]
    atoms.symbols[fixed_conc[:20]] = "X"

    generator = MixedSwapFlip(atoms, fixed_conc, fixed_chem_pot, ["O", "C"])

    mc = Montecarlo(atoms, 10000, generator=generator)
    mc.run(2000)

    assert all(s in ["Si", "X"] for s in atoms.symbols[fixed_conc])
    assert all(s in ["O", "C"] for s in atoms.symbols[fixed_chem_pot])


def test_mc_reset_step_counter(db_name):
    atoms = get_example_mc_system(db_name)
    mc = Montecarlo(atoms, 200)

    assert mc.current_step == 0
    mc.run(5)
    assert mc.current_step == 5
    # We always reset on a new run
    mc.run(3)
    assert mc.current_step == 3
    # pylint: disable=protected-access
    mc._reset_internal_counters()
    assert mc.current_step == 0


def test_evaluator(example_system):
    atoms = example_system

    # Test initializing with atoms object
    mc = Montecarlo(atoms, 200)
    assert mc.atoms is atoms
    assert mc.atoms is mc.evaluator.atoms

    # Test initializing with evaluator object
    evaluator = CEMCEvaluator(atoms)
    mc = Montecarlo(evaluator, 200)
    assert mc.atoms is atoms
    assert mc.evaluator is evaluator


@pytest.mark.parametrize("rep", [(2, 1, 1), (3, 3, 3)])
@pytest.mark.parametrize("seed", [10, 42])
def test_default_swapmove_background(make_settings_with_bkg, rep, set_rng, seed):
    """Test that we ignore background indices in
    the default MC swap move object"""
    set_rng(seed)
    settings, atoms = make_settings_with_bkg(rep)

    bkg_indices = [at.index for at in atoms if at.symbol == "O"]
    assert settings.background_indices == bkg_indices

    mc = Montecarlo(atoms, 300)

    assert mc.generator.indices == settings.non_background_indices

    gen = mc.generator

    # Ensure none of the background indices are tracked
    for index in settings.background_indices:
        assert not gen.is_tracked(index)

    # Ensure all non-background indices are tracked
    for index in settings.non_background_indices:
        assert gen.is_tracked(index)

    # Verify we can run
    mc.run(50)

    # Make an MC with no background "protection"
    swapmove = RandomSwap(atoms, indices=None)
    mc = Montecarlo(atoms, 300, generator=swapmove)

    with pytest.raises(RuntimeError):
        # We will try to swap a background index, which will fail
        mc.run(100)


def test_mc_irun(example_system):
    mc = Montecarlo(example_system, 300)

    it = mc.irun(20)
    assert mc.current_step == 0

    for _ in range(10):
        next(it)
    assert mc.current_step == 10

    for _ in it:
        pass
    assert mc.current_step == 20

    # Check that we reset everything
    it = mc.irun(20)
    assert mc.current_step == 0
    for i, _ in enumerate(it):
        assert mc.current_step == i + 1
    assert mc.current_step == 20


@pytest.mark.parametrize("temp", [1, 300, 30_000])
def test_mc_step(example_system, temp, make_tempfile):
    """Test the contents of the MCStep"""
    file = make_tempfile("mc_step.json")
    mc = Montecarlo(example_system, temp)

    nsteps = 50
    it = mc.irun(nsteps)
    assert mc.current_step == 0

    # Check we can run for a bit, and stop to inspect
    step0 = next(it)
    for ii, step in enumerate(it, start=2):
        assert isinstance(step, MCStep)
        assert step.step == ii
        # Verify the contents of the MCStep
        assert step.move_accepted is True or step.move_accepted is False
        assert len(step.last_move) == 2
        for move in step.last_move:
            assert isinstance(move, SystemChange)
        assert isinstance(step.energy, float)
        assert step.energy == mc.current_energy
        for change in step.last_move:
            # Find the expected symbol, based on whether the move was accepted
            exp_symb = change.new_symb if step.move_accepted else change.old_symb
            assert mc.atoms.symbols[change.index] == exp_symb, ii
        # Test saving/loading a step
        step.save(file)
        loaded = MCStep.load(file)
        assert step == loaded
    assert step0.step == 1  # Check we didn't mutate the original step
    assert mc.current_step == nsteps
    assert step.step == nsteps
    assert step.energy == mc.current_energy


def test_on_temp_change(example_system, mocker):
    mc = Montecarlo(example_system, 300)
    spy = mocker.spy(mc, "_on_temp_change")
    mc.run(5)
    assert spy.call_count == 0
    assert mc.mean_energy.mean != 0
    assert mc.mean_energy._n_samples > 0
    assert mc.energy_squared != 0
    # Change the temperature, and verify averagers are reset.
    mc.temperature = 400
    assert spy.call_count == 1
    assert mc.mean_energy.mean == 0
    assert mc.mean_energy._n_samples == 0
    assert mc.energy_squared.mean == 0
    # Verify it's also called when using the old attribute name
    mc.T = 200
    assert spy.call_count == 2


def test_call_observers(example_system, mocker):
    mc = Montecarlo(example_system, 300)
    obs = DummyObs()
    mc.attach(obs)

    mc.run(10)
    assert obs.call_count == 10

    mc.run(10, call_observers=False)
    assert obs.call_count == 10

    mc.run(20)
    assert obs.call_count == 30

    for _ in mc.irun(10, call_observers=False):
        pass
    assert obs.call_count == 30

    for _ in mc.irun(10):
        pass
    assert obs.call_count == 40


def test_set_temp_kT(example_system):
    mc = Montecarlo(example_system, 300)
    # Check both the temperature and kT properties are correctly set.
    assert mc.temperature == pytest.approx(300)
    assert mc.kT == pytest.approx(300.0 * kB)

    # Set the temperature, and verify the value
    for temp in np.random.uniform(0, 4_000, size=15):
        mc.temperature = temp
        assert mc.temperature == pytest.approx(temp)
        assert mc.kT == pytest.approx(temp * kB)
