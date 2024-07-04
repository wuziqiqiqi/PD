import pytest
from cleases.montecarlo.observers import ConcentrationObserver
from cleases.datastructures import SystemChange, MCStep, SystemChanges
from ase.build import bulk


@pytest.fixture
def make_step():
    def _make_step(**kwargs):
        input = {
            "step": 0,
            "energy": -1.5,
            "move_accepted": True,
            "last_move": [],
        }
        input.update(**kwargs)
        return MCStep(**input)

    return _make_step


def test_update(make_step):
    atoms = bulk("Au") * (7, 7, 7)
    obs = ConcentrationObserver(atoms, element="Au")
    assert obs.current_conc == pytest.approx(1.0)

    changes = [
        SystemChange(0, "Au", "Cu"),
        SystemChange(1, "Au", "Cu"),
        SystemChange(2, "Au", "Cu"),
    ]
    step = make_step(last_move=changes)
    obs.observe_step(step)
    N = len(atoms)
    assert obs.current_conc == pytest.approx(1.0 - 3.0 / N)

    changes = [SystemChange(0, "Cu", "Au")]
    step = make_step(last_move=changes)
    obs.observe_step(step)
    assert obs.current_conc == pytest.approx(1.0 - 2.0 / N)

    # Move was rejected, no change to the concentration
    changes = [SystemChange(0, "Cu", "Au")]
    step = make_step(last_move=changes, move_accepted=False)
    obs.observe_step(step)
    assert obs.current_conc == pytest.approx(1.0 - 2.0 / N)


def test_peak(make_step):
    atoms = bulk("Au") * (7, 7, 7)
    obs = ConcentrationObserver(atoms, element="Au")
    assert obs.current_conc == pytest.approx(1.0)

    changes = [
        SystemChange(0, "Au", "Cu"),
        SystemChange(1, "Au", "Cu"),
        SystemChange(2, "Au", "Cu"),
    ]
    step = make_step(last_move=changes)
    new_conc = obs.observe_step(step, peak=True)
    assert new_conc == pytest.approx(1.0 - 3.0 / len(atoms))
    assert obs.current_conc == pytest.approx(1.0)


def test_reset(make_step):
    atoms = bulk("Au") * (7, 7, 7)
    obs = ConcentrationObserver(atoms, element="Au")
    assert obs.current_conc == pytest.approx(1.0)

    changes = [
        SystemChange(0, "Au", "Cu"),
        SystemChange(1, "Au", "Cu"),
        SystemChange(2, "Au", "Cu"),
    ]
    step = make_step(last_move=changes)
    obs.observe_step(step)
    avg = obs.get_averages()
    expect = (1.0 + 1.0 - 3.0 / len(atoms)) / 2
    assert avg["conc_Au"] == pytest.approx(expect)

    obs.reset()
    avg = obs.get_averages()
    assert avg["conc_Au"] == pytest.approx(1.0 - 3.0 / len(atoms))


def test_rejected(make_step):
    atoms = bulk("Au") * (7, 7, 7)
    obs = ConcentrationObserver(atoms, element="Au")
    changes = [
        SystemChange(0, "Au", "Cu"),
        SystemChange(1, "Au", "Cu"),
        SystemChange(2, "Au", "Cu"),
    ]
    step = make_step(move_accepted=False, last_move=changes)
    conc = obs.observe_step(step)
    # Move was rejected, so no change should be seen in the concentration
    assert conc == pytest.approx(1.0)
