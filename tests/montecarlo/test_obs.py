import pytest
import ase
import numpy as np
from cleases.settings import Concentration, CEBulk
from cleases.calculator import attach_calculator
from cleases.corr_func import CorrFunction
from cleases.montecarlo import Montecarlo, observers


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


@pytest.fixture
def example_mc(example_system):
    return Montecarlo(example_system, 300)


@pytest.mark.parametrize("temp", [1, 100_000])
def test_move_observer(example_mc: Montecarlo, temp):
    atoms = example_mc.atoms

    obs = observers.MoveObserver(atoms, only_accept=False)
    example_mc.attach(obs)
    example_mc.temperature = temp

    N_runs = 200
    example_mc.run(N_runs)

    images = obs.reconstruct()
    assert len(images) == N_runs
    assert len(obs.steps) == N_runs
    prev = images[0]
    for ii, (atoms, step) in enumerate(zip(images, obs.steps)):
        if ii == 0:
            prev = atoms.copy()
            continue
        assert isinstance(atoms, ase.Atoms)
        assert atoms.get_potential_energy() == step.energy
        changes = step.last_move
        if step.move_accepted:
            # Some symbols should've changed
            for change in changes:
                assert atoms.symbols[change.index] == change.new_symb
        else:
            # All symbols should be the same
            assert (prev.symbols == atoms.symbols).all()
        assert np.allclose(prev.get_positions(), atoms.get_positions())
        prev = atoms.copy()

    obs.reset()
    assert len(obs.steps) == 0


@pytest.mark.parametrize("temp", [10, 100_000])
def test_move_obs_live(example_mc, temp):
    atoms = example_mc.atoms

    obs = observers.MoveObserver(atoms, only_accept=False)
    example_mc.attach(obs)
    example_mc.temperature = temp

    for step in example_mc.irun(200):
        # Get the last reconstructed image
        for reconstructed in obs.reconstruct_iter():
            pass
        assert (reconstructed.symbols == atoms.symbols).all()
        if not step.move_accepted:
            for change in step.last_move:
                assert atoms.symbols[change.index] == change.old_symb, change
        else:
            for change in step.last_move:
                assert atoms.symbols[change.index] == change.new_symb, change


def test_move_observer_only_accept(example_mc: Montecarlo):
    atoms = example_mc.atoms

    obs = observers.MoveObserver(atoms, only_accept=True)
    example_mc.attach(obs)
    example_mc.temperature = 1

    N_runs = 200
    for step in example_mc.irun(N_runs):
        if step.move_accepted:
            for reconstructed in obs.reconstruct_iter():
                pass
            assert (reconstructed.symbols == atoms.symbols).all()
            for change in step.last_move:
                assert atoms.symbols[change.index] == change.new_symb, change

    assert all(step.move_accepted for step in obs.steps)
    assert len(obs.steps) < N_runs

    images = obs.reconstruct()
    assert len(images) == len(obs.steps)


def test_move_obs_bad_interval(example_mc: Montecarlo):
    obs = observers.MoveObserver(example_mc.atoms)
    for interval in [2, -1, 3, 0, 10]:
        with pytest.raises(ValueError):
            example_mc.attach(obs, interval=interval)
    example_mc.attach(obs, interval=1)  # The only OK interval
    assert len(example_mc.observers) == 1
