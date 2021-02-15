import os
from pathlib import Path
import json
import pytest
import numpy as np
from ase.build import bulk
from ase.geometry import get_layers
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
from clease.montecarlo.observers import CorrelationFunctionObserver
from clease.montecarlo.observers import Snapshot
from clease.montecarlo.observers import EnergyEvolution
from clease.montecarlo.observers import SiteOrderParameter
from clease.montecarlo.observers import LowestEnergyStructure
from clease.montecarlo.observers import DiffractionObserver, AcceptanceRate
from clease.montecarlo.constraints import ConstrainSwapByBasis, FixedElement
from clease.montecarlo import RandomSwap, MixedSwapFlip
from clease.settings import CEBulk, Concentration
from clease.corr_func import CorrFunction
from clease.tools import SystemChange

# Set the random seed
np.random.seed(0)

almgsix_eci_file = Path(__file__).parent / 'almgsix_eci.json'


@pytest.fixture
def almgsix_eci():
    with almgsix_eci_file.open() as file:
        return json.load(file)


def get_example_mc_system(db_name):
    conc = Concentration(basis_elements=[['Au', 'Cu']])
    settings = CEBulk(db_name=db_name,
                      concentration=conc,
                      crystalstructure='fcc',
                      a=4.0,
                      max_cluster_size=3,
                      max_cluster_dia=[5.0, 4.1],
                      size=[2, 2, 2])

    atoms = settings.atoms.copy() * (3, 3, 3)
    # Insert a few different symbols
    atoms.symbols = 'Au'
    atoms.symbols[:10] = 'Cu'
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci['c0'] = -1.0
    eci['c2_d0000_0_00'] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    return atoms


def get_rocksalt_mc_system(db_name):
    conc = Concentration(basis_elements=[['Si', 'X'], ['O', 'C']])
    settings = CEBulk(db_name=db_name,
                      concentration=conc,
                      crystalstructure='rocksalt',
                      a=4.0,
                      max_cluster_size=3,
                      max_cluster_dia=[2.51, 3.0],
                      size=[2, 2, 2])
    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci['c0'] = -1.0
    eci['c2_d0000_0_00'] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    return atoms


@pytest.mark.slow
def test_run_heavy(db_name):
    conc = Concentration(basis_elements=[['Au', 'Cu']])
    settings = CEBulk(db_name=db_name,
                      concentration=conc,
                      crystalstructure='fcc',
                      a=4.0,
                      max_cluster_size=4)

    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci['c0'] = -1.0
    eci['c2_d0000_0_00'] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    # Insert a few elements
    for i in range(10):
        atoms[i].symbol = 'Cu'

    E = []
    for T in [1000, 500, 100]:
        mc = Montecarlo(atoms, T)
        mc.run(steps=10000)
        E.append(mc.get_thermodynamic_quantities()['energy'])

    cf = CorrFunction(atoms.calc.settings)
    cf_calc = atoms.calc.get_cf()
    cf_scratch = cf.get_cf(atoms)

    print(len(cf_scratch))

    os.remove(db_name)
    for k, v in cf_scratch.items():
        assert v == pytest.approx(cf_calc[k])

    # Make sure that the energies are decreasing by comparing
    # the first and last
    assert E[0] >= E[-1]


def test_run(db_name):
    atoms = get_example_mc_system(db_name)

    E = []
    for T in [1000, 500, 100]:
        mc = Montecarlo(atoms, T)
        mc.run(steps=10000)
        E.append(mc.get_thermodynamic_quantities()['energy'])

    cf = CorrFunction(atoms.calc.settings)
    cf_calc = atoms.calc.get_cf()
    cf_scratch = cf.get_cf(atoms)

    os.remove(db_name)
    for k, v in cf_scratch.items():
        assert v == pytest.approx(cf_calc[k])

    # Make sure that the energies are decreasing by comparing
    # the first and last
    assert E[0] >= E[-1]


def test_corr_func_observer(db_name):
    atoms = get_example_mc_system(db_name)

    atoms[0].symbol = 'Cu'
    atoms[1].symbol = 'Cu'

    mc = Montecarlo(atoms, 600)
    obs = CorrelationFunctionObserver(atoms.calc)
    mc.attach(obs, interval=1)
    mc.run(steps=1000)
    thermo = mc.get_thermodynamic_quantities()
    _ = obs.get_averages()
    assert obs.counter == 1001

    cf_keys = atoms.calc.get_cf().keys()

    for k in cf_keys:
        assert 'cf_' + k in thermo.keys()


def test_snapshot(db_name, make_tempfile):
    atoms = get_example_mc_system(db_name)

    fname = make_tempfile('snapshot.traj')
    obs = Snapshot(atoms, fname=fname)

    mc = Montecarlo(atoms, 600)
    mc.attach(obs, interval=100)
    mc.run(steps=1000)
    assert len(obs.traj) == 10

    # Test extension-less
    fname = make_tempfile('snapshot')
    obs = Snapshot(atoms, fname=fname)
    assert str(obs.fname) == fname + '.traj'


def test_energy_evolution(db_name, make_tempfile):
    print(db_name)
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
    fname_base = 'energy_evol_no_ext'
    fname = make_tempfile(fname_base)
    obs.save(fname=fname)
    assert os.path.isfile(str(fname) + '.csv')
    assert not os.path.isfile(fname)


def test_site_order_parameter(db_name):
    atoms = get_example_mc_system(db_name)

    # Manually change some symbols
    atoms.symbols = 'Au'
    atoms.symbols[:3] = 'Cu'

    obs = SiteOrderParameter(atoms)
    mc = Montecarlo(atoms, 600)
    mc.attach(obs)
    mc.run(steps=1000)
    avg = obs.get_averages()
    assert avg['site_order_average'] <= 6.0

    thermo = mc.get_thermodynamic_quantities()

    assert 'site_order_average' in thermo.keys()
    assert 'site_order_std' in thermo.keys()


def test_lowest_energy_obs(db_name):
    atoms = get_example_mc_system(db_name)

    atoms[0].symbol = 'Cu'
    atoms[1].symbol = 'Cu'
    atoms[2].symbol = 'Cu'

    low_en = LowestEnergyStructure(atoms)

    mc = Montecarlo(atoms, 700)
    energy_evol = EnergyEvolution(mc)
    mc.attach(low_en, interval=1)
    mc.attach(energy_evol, interval=1)

    mc.run(steps=1000)
    assert np.min(energy_evol.energies) == pytest.approx(low_en.lowest_energy)
    assert low_en.emin_atoms.get_potential_energy() == pytest.approx(low_en.lowest_energy)


def test_diffraction_obs(db_name):
    atoms = get_example_mc_system(db_name)

    atoms[0].symbol = 'Cu'
    atoms[1].symbol = 'Cu'
    atoms[2].symbol = 'Cu'

    obs = DiffractionObserver(atoms=atoms,
                              k_vector=[0.25, 0.0, 0.0],
                              name='reflect1',
                              active_symbols=['Cu'],
                              all_symbols=['Au', 'Cu'])

    mc = Montecarlo(atoms, 600)
    mc.attach(obs)

    mc.run(steps=1000)
    thermo = mc.get_thermodynamic_quantities()

    assert 'reflect1' in thermo.keys()


def test_constrain_swap(db_name):
    atoms = get_rocksalt_mc_system(db_name)
    settings = atoms.calc.settings
    i_by_basis = settings.index_by_basis

    # Insert a few vacancies
    num_X = 0
    for atom in atoms:
        if atom.symbol == 'Si':
            atom.symbol = 'X'
            num_X += 1

        if num_X >= 20:
            break

    # Insert a few C
    num_C = 0
    for atom in atoms:
        if atom.symbol == 'O':
            atom.symbol = 'C'
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

    allowed_elements = [['Si', 'X'], ['O', 'C']]
    for basis, allowed in zip(i_by_basis, allowed_elements):
        for indx in basis:
            assert atoms[indx].symbol in allowed


def test_fixed_element(db_name):
    atoms = get_rocksalt_mc_system(db_name)

    # Insert a few vacancies
    num_X = 0
    for atom in atoms:
        if atom.symbol == 'Si':
            atom.symbol = 'X'
            num_X += 1

        if num_X >= 20:
            break

    # Insert a few C
    num_C = 0
    for atom in atoms:
        if atom.symbol == 'O':
            atom.symbol = 'C'
            num_C += 1

        if num_C >= 20:
            break

    # Let's say that all Si atoms should be fixed
    fixed_element = FixedElement('Si')
    generator = RandomSwap(atoms)
    generator.add_constraint(fixed_element)

    mc = Montecarlo(atoms, 10000, generator=generator)

    si_indices = [atom.index for atom in atoms if atom.symbol == 'Si']
    mc.run(steps=1000)

    si_after = [atom.index for atom in atoms if atom.symbol == 'Si']
    assert si_indices == si_after


def test_fixed_indices(db_name):
    atoms = get_rocksalt_mc_system(db_name)

    # Insert a few vacancies
    num_X = 0
    for atom in atoms:
        if atom.symbol == 'Si':
            atom.symbol = 'X'
            num_X += 1

        if num_X >= 20:
            break

    # Insert a few C
    num_C = 0
    for atom in atoms:
        if atom.symbol == 'O':
            atom.symbol = 'C'
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
    conc = Concentration(basis_elements=[['Al', 'Mg', 'Si', 'X']])
    settings = CEBulk(conc,
                      crystalstructure='fcc',
                      a=4.05,
                      size=[1, 1, 1],
                      max_cluster_size=3,
                      max_cluster_dia=[5.0, 5.0],
                      db_name=db_name)
    settings.basis_func_type = "binary_linear"

    atoms = bulk('Al', a=4.05, cubic=True) * (2, 2, 2)
    atoms = attach_calculator(settings, atoms, almgsix_eci)
    expect = atoms.copy()
    layers, _ = get_layers(expect, (1, 0, 0))
    for i, layer in enumerate(layers):
        if layer % 2 == 0:
            expect[i].symbol = 'Si'
        else:
            expect[i].symbol = 'Mg'

    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = 'Mg'
        atoms[i + int(len(atoms) / 2)].symbol = 'Si'

    mc = Montecarlo(atoms, 1000)
    print(mc.current_energy)
    en_obs = EnergyEvolution(mc)
    mc.attach(en_obs)
    temps = [1000, 800, 600, 500, 400, 300, 200, 100]
    for T in temps:
        mc.T = T
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
        acc_rate([SystemChange(0, 'Al', 'Mg', '')])

    assert acc_rate.rate == pytest.approx(1.0)

    num_reject = 3
    for _ in range(num_reject):
        acc_rate([SystemChange(0, 'Al', 'Al', '')])
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
    fixed_conc = [atom.index for atom in atoms if atom.symbol == 'Si']
    fixed_chem_pot = [atom.index for atom in atoms if atom.symbol == 'O']
    atoms.symbols[fixed_conc[:20]] = 'X'

    generator = MixedSwapFlip(atoms, fixed_conc, fixed_chem_pot, ['O', 'C'])

    mc = Montecarlo(atoms, 10000, generator=generator)
    mc.run(2000)

    assert all(s in ['Si', 'X'] for s in atoms.symbols[fixed_conc])
    assert all(s in ['O', 'C'] for s in atoms.symbols[fixed_chem_pot])


def test_mc_reset_step_counter(db_name):
    atoms = get_example_mc_system(db_name)
    mc = Montecarlo(atoms, 200)

    assert mc.current_step == 0
    mc.run(5)
    assert mc.current_step == 5
    # If we didn't reset step counter,
    # we couldn't be able to get a smaller step count
    mc.run(3)
    assert mc.current_step == 3
    # pylint: disable=protected-access
    mc._reset_internal_counters()
    assert mc.current_step == 0
