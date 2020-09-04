import os
import pytest
import numpy as np
from pathlib import Path
import json
from ase.build import bulk
from ase.geometry import get_layers
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo
from clease.montecarlo.observers import CorrelationFunctionObserver
from clease.montecarlo.observers import Snapshot
from clease.montecarlo.observers import EnergyEvolution
from clease.montecarlo.observers import SiteOrderParameter
from clease.montecarlo.observers import LowestEnergyStructure
from clease.montecarlo.observers import DiffractionObserver
from clease.montecarlo.constraints import ConstrainSwapByBasis
from clease.montecarlo.constraints import FixedElement
from clease.settings import CEBulk, Concentration
from clease.corr_func import CorrFunction

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


def test_snapshot(db_name, tmpdir):
    atoms = get_example_mc_system(db_name)

    atoms[0].symbol = 'Cu'
    atoms[1].symbol = 'Cu'

    obs = Snapshot(fname=str(tmpdir / 'snapshot'), atoms=atoms)

    mc = Montecarlo(atoms, 600)
    mc.attach(obs, interval=100)
    mc.run(steps=1000)
    assert len(obs.traj) == 10
    try:
        fname = tmpdir / 'snapshot.traj'
        os.remove(fname)
    except OSError:
        pass


def test_energy_evolution(db_name, tmpdir):
    atoms = get_example_mc_system(db_name)
    atoms[0].symbol = 'Cu'
    atoms[1].symbol = 'Cu'

    mc = Montecarlo(atoms, 600)
    obs = EnergyEvolution(mc)
    mc.attach(obs, interval=50)
    mc.run(steps=1000)

    # Just confirm that the save function works
    fname = 'energy_evol'
    obs.save(fname=str(tmpdir / fname))
    file = tmpdir / (fname + '.csv')
    assert file.exists()
    try:
        file.remove()
    except OSError:
        pass

    # Check the number of energy values
    assert len(obs.energies) == 20


def test_site_order_parameter(db_name):
    atoms = get_example_mc_system(db_name)

    atoms[0].symbol = 'Cu'
    atoms[1].symbol = 'Cu'
    atoms[2].symbol = 'Cu'

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

    mc = Montecarlo(atoms, 600)
    mc.add_constraint(cnst)
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

    mc = Montecarlo(atoms, 10000)
    mc.add_constraint(fixed_element)

    si_indices = [atom.index for atom in atoms if atom.symbol == 'Si']
    mc.run(steps=1000)

    si_after = [atom.index for atom in atoms if atom.symbol == 'Si']
    assert si_indices == si_after


@pytest.mark.slow
def test_gs_mgsi(db_name, almgsix_eci):
    conc = Concentration(basis_elements=[['Al', 'Mg', 'Si', 'X']])
    settings = CEBulk(conc, crystalstructure='fcc', a=4.05,
                      size=[1, 1, 1], max_cluster_size=3,
                      max_cluster_dia=[5.0, 5.0])
    settings.basis_func_type = "binary_linear"

    atoms = bulk('Al', a=4.05, cubic=True)*(2, 2, 2)
    atoms = attach_calculator(settings, atoms, almgsix_eci)
    expect = atoms.copy()
    layers, _ = get_layers(expect, (1, 0, 0))
    for i in range(len(layers)):
        if layers[i] % 2 == 0:
            expect[i].symbol = 'Si'
        else:
            expect[i].symbol = 'Mg'

    for i in range(int(len(atoms)/2)):
        atoms[i].symbol = 'Mg'
        atoms[i+int(len(atoms)/2)].symbol = 'Si'

    mc = Montecarlo(atoms, 1000)
    print(mc.current_energy)
    en_obs = EnergyEvolution(mc)
    mc.attach(en_obs)
    temps = [1000, 800, 600, 500, 400, 300, 200, 100]
    for T in temps:
        mc.T = T
        mc.run(steps=100*len(atoms))

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
