from json import load
import pytest
import numpy as np
from cleases.datastructures import SystemChange, MCStep
from cleases.jsonio import read_json


@pytest.fixture
def swap_change():
    change1 = SystemChange(0, "A", "B")
    change2 = SystemChange(1, "B", "A")
    return [change1, change2]


def test_step_eq_and_order(swap_change):
    step1 = MCStep(0, 1.0, True, swap_change, other={"foo": "bar"})
    step2 = MCStep(0, 1.0, True, tuple(swap_change))

    assert step1.other != pytest.approx(step2.other)
    # "other" isn't included in the comparison.
    assert step1 == step2
    step1.energy = 1.5
    assert step1 != step2

    # Verify that we cannot order steps
    with pytest.raises(TypeError):
        step1 < step2
    with pytest.raises(TypeError):
        step1 > step2


def test_save_load(swap_change, make_tempfile):
    file = make_tempfile("step.json")
    step = MCStep(0, 1.0, True, swap_change)
    step.other["test"] = np.array([1, 2, 3.0])
    step.save(file)
    loaded = read_json(file)
    assert step is not loaded
    assert step == loaded
    assert "test" in step.other
    assert np.allclose(step.other["test"], loaded.other["test"])


@pytest.mark.parametrize("move_accepted", [True, False])
def test_move_rejected(move_accepted):
    step = MCStep(0, 1.0, move_accepted, [])
    assert step.move_rejected is (not move_accepted)
