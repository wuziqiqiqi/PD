from json import load
import pytest
from clease.datastructures import SystemChange, MCStep
from clease.jsonio import read_json


@pytest.fixture
def swap_change():
    change1 = SystemChange(0, "A", "B")
    change2 = SystemChange(1, "B", "A")
    return [change1, change2]


def test_step_eq_and_order(swap_change):
    step1 = MCStep(0, 1.0, True, swap_change)
    step2 = MCStep(0, 1.0, True, tuple(swap_change))

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
    step.save(file)
    loaded = read_json(file)
    assert step is not loaded
    assert step == loaded
