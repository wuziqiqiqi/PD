import pytest
from ase import Atoms
from clease.montecarlo.observers import Snapshot


@pytest.mark.parametrize("mode", ["w", "a"])
def test_snapshot(traj_file, mode):
    snap = Snapshot(Atoms(), fname=traj_file, mode=mode)
    snap([])
    assert len(snap.traj) == 1

    snap([])
    assert len(snap.traj) == 2

    snap.reset()

    # Reset should have no effect
    assert len(snap.traj) == 2

    # Re-open a new observer with the same file name
    snap = Snapshot(Atoms(), fname=traj_file, mode=mode)
    if mode == "a":
        assert len(snap.traj) == 2
    elif mode == "w":
        assert len(snap.traj) == 0
