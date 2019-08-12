from clease.montecarlo.constraints import MCConstraint
import numpy as np


class ConstrainSwapByBasis(MCConstraint):
    """
    Constraint that restricts swaps of atoms within a given basis.
    This constraint is intended to be used togerthing with canonical
    Monte Carlo calculations where the trial moves consist of swapping
    two atoms.

    Parameters:

    atoms: Atoms
        Atoms object used in the MC simulation

    index_by_basis: list
        Indices ordered by basis (can be taken from ClusterExpansionSetting)
        If an atoms object has 10 sites, and the first 4 belongs to one basis,
        the three next belongs to one basis and the last three belongs to one
        basis, this argument would be
        [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

        Swaps are only allowed within each basis, not across two basis.
    """
    def __init__(self, atoms, index_by_basis):
        self.basis = np.zeros(len(atoms))

        for i, indices in enumerate(index_by_basis):
            for x in indices:
                self.basis[x] = i

    def __call__(self, system_changes):
        if len(system_changes) != 2:
            raise ValueError("ConstrainSwapByBasis requires swap move")

        i1 = system_changes[0][0]
        b1 = self.basis[i1]
        i2 = system_changes[1][0]
        b2 = self.basis[i2]
        return b1 == b2
