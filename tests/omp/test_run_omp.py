import random
from collections import defaultdict
import pytest
import numpy as np
from cleases.settings import CEBulk, Concentration
from cleases.calculator.util import attach_calculator
from cleases.montecarlo import Montecarlo

# Require the --openmp mark
pytestmark = pytest.mark.openmp


@pytest.fixture
def settings(db_name):
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    return CEBulk(conc, crystalstructure="fcc", a=4.05, size=(1, 1, 1), db_name=db_name)


@pytest.fixture
def eci(settings):
    all_cf_names = settings.all_cf_names
    return {name: random.uniform(-1, 1) for name in all_cf_names}


@pytest.fixture
def atoms(settings):
    ats = settings.prim_cell * (5, 5, 5)
    N_half = len(ats) // 2
    ats.symbols[:N_half] = "Au"
    ats.symbols[N_half:] = "Cu"
    syms = list(ats.symbols)
    random.shuffle(syms)
    ats.symbols = syms
    return ats


@pytest.mark.parametrize("temp", [1, 2_000, 10_000, 100_000])
def test_run_mc_with_threads(atoms, temp, settings, eci, set_rng, cpu_count):
    quantities = defaultdict(list)

    cores = [1, 1, 2]  # At the very least test these cores.
    for i in range(3, cpu_count + 1):
        cores.append(i)
    base_atoms = atoms.copy()

    # Reuse the same atoms, just reset symbols.
    atoms_with_calc = attach_calculator(settings, atoms, eci)
    calc = atoms_with_calc.calc

    for num_threads in cores:
        set_rng()
        atoms_with_calc.symbols[:] = base_atoms.symbols
        mc = Montecarlo(atoms_with_calc, temp)
        mc.atoms.calc.set_num_threads(num_threads)
        # Verify that the C++ object has the right number of threads.
        assert calc.updater.get_num_threads() == num_threads
        mc.run(steps=150)

        # Fetch some quantities we want to measure
        quantities["current_energy"].append(mc.current_energy)
        quantities["avg_en"].append(mc.mean_energy.mean)
        quantities["en_sq"].append(mc.energy_squared.mean)
        quantities["num_accept"].append(mc.num_accepted)
        thermo = mc.get_thermodynamic_quantities()
        quantities["heat_cap"].append(thermo["heat_capacity"])
        quantities["energy_var"].append(thermo["energy_var"])

    for name, qty in quantities.items():
        assert np.allclose(qty[0], qty), name


def test_attach_num_threads(settings, eci):
    base_atoms = settings.prim_cell.copy()
    for threads in range(1, 20):
        atoms = attach_calculator(settings, base_atoms, eci, num_threads=threads)
        assert atoms.calc.get_num_threads() == threads
        # Verify the C++ updater agrees with the number of threads.
        assert atoms.calc.updater.get_num_threads() == threads
