import pytest
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from cleases.calculator import attach_calculator
from cleases.montecarlo import SGCMonteCarlo
from cleases.montecarlo.montecarlo import Montecarlo
from cleases.montecarlo.observers import SGCState, MultiStateSGCConcObserver, ConcentrationObserver
from cleases.montecarlo.observers.mc_observer import MCObserver
from cleases.settings import CEBulk, Concentration
from cleases.corr_func import CorrFunction
from cleases.montecarlo.constraints import ConstrainElementInserts
from cleases.montecarlo.constraints import PairConstraint
from cleases.montecarlo import RandomFlip
from cleases.tools import species_chempot2eci


def check_obs_is_attached(mc: Montecarlo, obs: MCObserver):
    """Helper function to check that a given observer is attached.
    Raises an assertion error if its missing."""
    for o in mc.iter_observers():
        if obs is o:
            # We found the correct observer
            return
    assert False, "Missing observer!"


@pytest.fixture
def settings(db_name):
    """Example settings for an AuCu system for running MC"""
    conc = Concentration(basis_elements=[["Au", "Cu"]])
    settings_ = CEBulk(
        db_name=db_name,
        concentration=conc,
        crystalstructure="fcc",
        a=4.0,
        max_cluster_dia=[5.0, 4.1],
        size=[2, 2, 2],
    )
    return settings_


@pytest.fixture
def example_sgc_mc(settings, make_random_eci):
    def _make_example(temp=30_000, n=3, chem_pot=None, **kwargs):
        atoms = settings.prim_cell * (n, n, n)

        eci = make_random_eci(settings)

        atoms = attach_calculator(settings, atoms, eci)
        mc = SGCMonteCarlo(atoms, temp, symbols=["Au", "Cu"], **kwargs)
        return mc

    return _make_example


def test_run(settings):
    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}

    eci["c0"] = -1.0
    eci["c2_d0000_0_00"] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    E = []
    for T in [5000, 2000, 1000, 500, 100]:
        mc = SGCMonteCarlo(atoms, T, symbols=["Au", "Cu"])
        mc.run(steps=10000, chem_pot={"c1_0": -0.02})
        E.append(mc.get_thermodynamic_quantities()["energy"])

    cf_calc = atoms.calc.get_cf()
    cf_scratch = cf.get_cf(atoms)

    for k, v in cf_calc.items():
        assert v == pytest.approx(cf_calc[k])

    # Make sure that the energies are decreasing
    for i in range(1, len(E)):
        assert E[i - 1] >= E[i]


def test_conc_obs_sgc(settings):
    atoms = settings.atoms.copy() * (3, 3, 3)

    settings.set_active_template(atoms)
    eci = {k: 0.0 for k in settings.all_cf_names}
    eci["c0"] = -1.0
    eci["c2_d0000_0_00"] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    obs1 = ConcentrationObserver(atoms, "Au")
    obs2 = ConcentrationObserver(atoms, "Cu")
    conc1_orig = obs1.get_averages()
    conc2_orig = obs2.get_averages()

    mc = SGCMonteCarlo(atoms, 5000, symbols=["Au", "Cu"])
    mc.attach(obs1)
    mc.attach(obs2)

    check_obs_is_attached(mc, obs1)
    check_obs_is_attached(mc, obs2)

    for T in [5000, 2000, 500]:
        mc.temperature = T
        mc.run(steps=10000, chem_pot={"c1_0": -0.02})

        conc1 = obs1.get_averages()
        conc2 = obs2.get_averages()
        assert conc1 != pytest.approx(conc1_orig)
        assert conc2 != pytest.approx(conc2_orig)

        c_au = conc1["conc_Au"]
        c_cu = conc2["conc_Cu"]
        assert 0 <= c_au <= 1
        assert 0 <= c_cu <= 1
        assert c_au + c_cu == pytest.approx(1)
        conc1_orig = conc1
        conc2_orig = conc2

        # Verify that the observer quantities are also in the thermodynamic quantities
        thermo = mc.get_thermodynamic_quantities()
        for k, v in conc1.items():
            assert thermo[k] == pytest.approx(v)
        for k, v in conc2.items():
            assert thermo[k] == pytest.approx(v)
    # Verify observers are still attached
    check_obs_is_attached(mc, obs1)
    check_obs_is_attached(mc, obs2)


def test_constrain_inserts(db_name):
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
    eci["c2_d0000_0_00"] = -0.2
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    generator = RandomFlip(["Si", "X", "O", "C"], atoms)
    elem_basis = [["Si", "X"], ["O", "C"]]
    cnst = ConstrainElementInserts(atoms, settings.index_by_basis, elem_basis)
    generator.add_constraint(cnst)
    mc = SGCMonteCarlo(atoms, 100000, generator=generator)

    chem_pot = {"c1_0": 0.0, "c1_1": -0.1, "c1_2": 0.1}
    orig_symbols = [a.symbol for a in atoms]
    mc.run(steps=1000, chem_pot=chem_pot)

    new_symb = [a.symbol for a in atoms]
    # Confirm that swaps have taken place
    assert not all(map(lambda x: x[0] == x[1], zip(orig_symbols, new_symb)))

    # Check that we only have correct entries
    for allowed, basis in zip(elem_basis, settings.index_by_basis):
        for index in basis:
            assert atoms[index].symbol in allowed


def test_pair_constraint(db_name):
    a = 4.0
    conc = Concentration(basis_elements=[["Si", "X"]])
    settings = CEBulk(
        db_name=db_name,
        concentration=conc,
        crystalstructure="fcc",
        a=a,
        max_cluster_dia=[3.9, 3.0],
        size=[2, 2, 2],
    )

    atoms = settings.atoms.copy() * (3, 3, 3)
    cf = CorrFunction(settings)
    cf_scratch = cf.get_cf(settings.atoms)
    eci = {k: 0.0 for k, v in cf_scratch.items()}
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)

    generator = RandomFlip(["Si", "X"], atoms)
    cluster = settings.cluster_list.get_by_name("c2_d0000_0")[0]
    cnst = PairConstraint(["X", "X"], cluster, settings.trans_matrix, atoms)
    generator.add_constraint(cnst)

    mc = SGCMonteCarlo(atoms, 100000000, generator=generator)

    nn_dist = a / np.sqrt(2.0)
    for _ in range(20):
        mc.run(10, chem_pot={"c1_0": 0.0})

        X_idx = [atom.index for atom in atoms if atom.symbol == "X"]

        if len(X_idx) <= 1:
            continue

        # Check that there are no X that are nearest neighbours
        dists = [atoms.get_distance(i1, i2) for i1 in X_idx for i2 in X_idx[i1 + 1 :]]
        assert all(d > 1.05 * nn_dist for d in dists)


def test_multi_state_sgc_obs(db_name):
    conc = Concentration([["Ag", "Pt"]])
    settings = CEBulk(
        conc,
        a=4.0,
        crystalstructure="sc",
        db_name=db_name,
        max_cluster_dia=[4.01],
        size=[4, 4, 4],
    )
    atoms = settings.atoms.copy()
    eci = {"c0": 0.0, "c1_0": 0.0, "c2_d0000_0_00": 0.1}

    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    T = 400.0
    sgc = SGCMonteCarlo(atoms, T, symbols=["Ag", "Pt"])

    chem_pots = [
        {
            "Ag": 0.0,
        },
        {
            "Ag": 0.5,
        },
        {
            "Ag": -0.5,
        },
    ]

    chem_pots = [species_chempot2eci(settings.basis_functions, c) for c in chem_pots]
    ref_pot = chem_pots[0]

    states = [SGCState(T, c) for c in chem_pots]
    observer = MultiStateSGCConcObserver(
        SGCState(T, ref_pot),
        states,
        atoms.calc,
    )

    # Seed python's random number generator and numpy's
    np.random.seed(0)

    sgc.attach(observer)

    check_obs_is_attached(sgc, observer)
    sgc.run(steps=2000, chem_pot=ref_pot)

    # Check consistency. By construction the Ag concentration varies in the same
    # direction as the chemical potential (high potential --> more Ag)
    avg = observer.get_averages()
    conc_center = avg["400K_c1_0plus0_singlet_c1_0"]
    conc_plus = avg["400K_c1_0plus500_singlet_c1_0"]
    conc_minus = avg["400K_c1_0minus500_singlet_c1_0"]

    assert conc_plus > conc_center
    assert conc_minus < conc_center


def test_sgc_temp_change(example_sgc_mc):
    mc = example_sgc_mc(temp=10_000, observe_singlets=True)

    obs = mc.averager

    assert obs.energy.mean == 0
    assert obs.energy_sq.mean == 0
    assert (obs.singlets == 0).all()
    assert obs.counter == 0

    # Verify the averager observer is attached.
    check_obs_is_attached(mc, obs)

    mc.run(100, chem_pot={"c1_0": 0.0})

    assert obs.energy.mean != pytest.approx(0)
    assert obs.energy_sq.mean > 0
    assert (obs.singlets != 0).any()
    assert obs.counter == 100

    # Trigger a temp change, should reset averagers
    mc.temperature = 10_000
    assert obs.energy.mean == 0
    assert obs.energy_sq.mean == 0
    assert (obs.singlets == 0).all()
    assert obs.counter == 0

    # Verify the averager observer is still attached.
    check_obs_is_attached(mc, obs)


def test_no_clease_calc():
    """Verify that the SGC MC class fails if no Clease calculator is attached"""
    atoms = bulk("Au") * (4, 4, 4)
    assert atoms.calc is None
    with pytest.raises(ValueError):
        SGCMonteCarlo(atoms, 300)
    atoms.calc = EMT()
    with pytest.raises(ValueError):
        SGCMonteCarlo(atoms, 300)
