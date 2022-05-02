from typing import Sequence
import numpy as np
from ase import Atoms
from clease.datastructures.system_changes import SystemChanges
from .mc_constraint import MCConstraint


class ConstrainSwapByBasis(MCConstraint):
    """
    Constraint that restricts swaps of atoms within a given basis.
    This constraint is intended to be used together with canonical
    Monte Carlo calculations where the trial moves consist of swapping
    two atoms.

    Parameters:

    atoms: Atoms object
        ASE Atoms object used in the MC simulation

    index_by_basis: List[List[int]]
        Indices ordered by basis (same as ``index_by_basis`` parameter in
        the :class:`~clease.settings.settings.ClusterExpansionSettings`
        settings object.).
        If an Atoms object has 10 sites where the first 4 belongs to the
        first basis, the next 3 belongs to the next basis and the last 3
        belongs to the last basis, the ``index_by_basis`` would be
        [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]].

    Note: swaps are only allowed within each basis, not across two basis.
    """

    def __init__(self, atoms: Atoms, index_by_basis: Sequence[Sequence[int]]):
        self.basis = np.zeros(len(atoms), dtype=int)

        for i, indices in enumerate(index_by_basis):
            for x in indices:
                self.basis[x] = i

    def __call__(self, system_changes: SystemChanges) -> bool:
        if len(system_changes) != 2:
            raise ValueError("ConstrainSwapByBasis requires swap move")

        i1 = system_changes[0].index
        b1 = self.basis[i1]
        i2 = system_changes[1].index
        b2 = self.basis[i2]
        return b1 == b2
