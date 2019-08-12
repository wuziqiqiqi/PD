from clease.montecarlo.constraints import MCConstraint
import numpy as np


class ConstrainElementInserts(MCConstraint):
    """
    Constrain elements insert by basis. This constraint is intended
    to be used together with SGCMonteCarlo

    atoms: Atoms
        Atoms object used in the simulation

    index_by_basis: list
        Indices ordered by basis. See ConstrainSwapByBasis.

    element_by_basis: list
        List of which elements are allowed in each basis. If we have two
        basis and Si and O is allowed in the fist and Si and C is allowed in
        the second, this argument would be [['Si', 'O'], ['Si', 'C']]
    """
    def __init__(self, atoms, index_by_basis, element_by_basis):
        self.basis = np.zeros(len(atoms), dtype=int)
        for i, indices in enumerate(index_by_basis):
            for x in indices:
                self.basis[x] = i

        num_basis = len(element_by_basis)
        unique_elem = set([x for item in element_by_basis for x in item])
        self.element_allowed = {x: np.zeros(num_basis, dtype=np.uint8)
                                for x in unique_elem}

        # Initialise the element lookup
        for i, elements in enumerate(element_by_basis):
            for elem in elements:
                self.element_allowed[elem][i] = 1

    def __call__(self, system_changes):
        for change in system_changes:
            indx = change[0]
            new_elem = change[2]
            basis = self.basis[indx]
            if self.element_allowed[new_elem][basis] == 0:
                return False
        return True
