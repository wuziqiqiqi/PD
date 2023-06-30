"""Module for testing the MC Evaluator class"""
import pytest
from clease.settings import CEBulk, Concentration
from clease.calculator import attach_calculator
from clease.corr_func import CorrFunction
from clease.montecarlo import construct_evaluator


@pytest.fixture
def aucu_system(db_name):
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


@pytest.fixture
def get_cf_scratch():
    def _get_cf_scratch(atoms):
        cf = CorrFunction(atoms.calc.settings)
        return cf.get_cf(atoms)

    return _get_cf_scratch


def test_syncronize(aucu_system, get_cf_scratch):
    atoms = aucu_system
    eva = construct_evaluator(atoms)

    cf = get_cf_scratch(atoms)

    assert atoms.symbols[0] == "Cu"
    atoms.symbols[0] = "Au"

    # We haven't done any syncronization yet
    # so CF's shouldn't have changed
    cf_calc = atoms.calc.get_cf()
    assert cf == pytest.approx(cf_calc)

    # Now we syncronize, i.e. update the CF to reflect the change
    eva.synchronize()
    cf_calc = atoms.calc.get_cf()
    assert cf != pytest.approx(cf_calc)

    # Undo the change
    atoms.symbols[0] = "Cu"
    eva.synchronize()
    cf_calc = atoms.calc.get_cf()
    assert cf == pytest.approx(cf_calc)
