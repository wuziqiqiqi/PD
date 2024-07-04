import os
import time
from pathlib import Path
import json
import random
import pytest
import ase
import numpy as np
import matplotlib.pyplot as plt
from cleases.montecarlo import Montecarlo
from cleases.settings import Concentration, CEBulk
from cleases.corr_func import CorrFunction
from cleases.calculator import attach_calculator, Clease
from clease_cxx import has_parallel


@pytest.fixture
def make_example_mc_system(db_name):
    def _make_example_mc_system(size=(3, 3, 3), mcd=(5.0, 4.1)) -> ase.Atoms:
        conc = Concentration(basis_elements=[["Au", "Cu"]])
        settings = CEBulk(
            db_name=db_name,
            concentration=conc,
            crystalstructure="fcc",
            a=4.0,
            max_cluster_dia=mcd,
        )

        atoms = settings.prim_cell.copy() * size
        # Insert a few different symbols
        N = len(atoms)
        atoms.symbols = "Au"
        atoms.symbols[: (N // 2)] = "Cu"
        # Randomize the symbols
        new_syms = list(atoms.symbols)
        random.shuffle(new_syms)
        atoms.symbols = new_syms

        cf = CorrFunction(settings)
        cf_scratch = cf.get_cf(settings.atoms)
        eci = {k: random.uniform(-2, 2) for k, v in cf_scratch.items()}
        atoms = attach_calculator(settings, atoms=atoms, eci=eci)
        return atoms

    return _make_example_mc_system


@pytest.fixture
def run_simple_mc(make_example_mc_system):
    def _runner(size=(3, 3, 3), steps=1000, temps=(1000, 500, 100)):
        atoms = make_example_mc_system(size=size)
        for T in temps:
            mc = Montecarlo(atoms, T)
            mc.run(steps=steps)

    return _runner


def test_run_mc(benchmark, run_simple_mc):
    benchmark(run_simple_mc, steps=10_000, size=(4, 4, 4))


@pytest.mark.openmp
def test_mc_omp(benchmark, make_example_mc_system):
    assert has_parallel()
    steps = 100_000

    def _runner():
        results = []
        for num_threads in [1, 2, 3, 4, None]:
            if num_threads is None:
                num_threads = os.cpu_count()
            atoms = make_example_mc_system(size=(10, 10, 10), mcd=(6.0, 6.0, 6.0))
            calc: Clease = atoms.calc
            calc.set_num_threads(num_threads)
            mc = Montecarlo(atoms, 3_000)
            t_start = time.perf_counter()
            mc.run(steps=steps)
            dt = time.perf_counter() - t_start
            results.append({"dt": dt, "num_threads": num_threads})
        return results

    results = benchmark.pedantic(_runner, rounds=1, iterations=1)
    # Dump the timings to a file
    pa = Path(__file__).parent
    with open(pa / "openmp_timing.json", "w") as file:
        json.dump(results, file)
    x = [r["num_threads"] for r in results]
    y = np.array([r["dt"] for r in results])
    y /= y[0]
    plt.plot(x, y)
    plt.xlabel("# Threads")
    plt.ylabel("Relative Runtime (s)")
    plt.title(f"Number of steps: {steps}")
    plt.show()
