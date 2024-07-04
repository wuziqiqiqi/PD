"""Module for testing individial MC constraints"""
from clease.montecarlo.constraints import FixedIndices
from clease.datastructures import SystemChange


def test_fixed_indices():
    def change(idx):
        # Helper function to make dummy system changes
        return SystemChange(idx, "", "", "")

    # Fix these indices
    indices = [1, 2, 3]
    cnst = FixedIndices(indices)

    assert all(isinstance(v, int) for v in cnst.fixed_basis)

    # index 1 is fixed
    changes = [change(1), change(8)]
    assert cnst(changes) is False
    # none of the indices are fixed
    changes[0] = change(5)
    assert cnst(changes) is True

    # Add a change with a fixed index again
    changes.append(change(3))
    assert cnst(changes) is False
