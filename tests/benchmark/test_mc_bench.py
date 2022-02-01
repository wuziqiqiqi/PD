import random
import pytest
import ase
from clease.montecarlo import Montecarlo
from clease.settings import Concentration, CEBulk
from clease.corr_func import CorrFunction
from clease.calculator import attach_calculator


@pytest.fixture
def make_example_mc_system(db_name):
    def _make_example_mc_system(size=(3, 3, 3)) -> ase.Atoms:
        conc = Concentration(basis_elements=[["Au", "Cu"]])
        settings = CEBulk(
            db_name=db_name,
            concentration=conc,
            crystalstructure="fcc",
            a=4.0,
            max_cluster_dia=[5.0, 4.1],
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
        eci = {k: 0.0 for k, v in cf_scratch.items()}
        eci["c0"] = 1.0
        eci["c2_d0000_0_00"] = 2.5
        eci["c3_d0000_0_000"] = 3.5
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
