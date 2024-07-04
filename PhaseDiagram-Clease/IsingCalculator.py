from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList
import numpy as np
import json

class IsingCalc(Calculator):
    """Calculator that calculates the total energy of a system by summing
    the energies of individual atoms.
    """

    implemented_properties = ['energy']

    def __init__(self, E0 = 0, h = 0, J = 0, cutoff = 3):
        """Initialize the calculator.

        Parameters:
        atoms: ASE Atoms object
            The system of atoms.
        atom_energies: list of float
            List of energies for each atom in the system.
        """
        Calculator.__init__(self)
        self.atoms = None
        self.E0 = E0
        self.h = h
        self.J = J
        self.cutoff = cutoff

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions']):
        """Calculate the total energy of the system by summing the energies
        of individual atoms.

        Parameters:
        atoms: ASE Atoms object, optional
            The system of atoms. If not provided, the atoms object passed
            to the constructor will be used.
        properties: list of str, optional
            List of properties to calculate. Should only contain 'energy'.
        system_changes: list of str, optional
            List of changes to the system that require the calculator to
            be rerun.

        Returns:
        dict
            Dictionary of calculated properties.
        """
        self.atoms = atoms
        E = self.E0

        nl = NeighborList(cutoffs=[self.cutoff / 2] * len(atoms),
            self_interaction=False,
            bothways=True)
        
        nl.update(atoms)
    
        convertedNumbers = np.array(self.atoms.numbers == 79, dtype=bool) * 2 - 1

        # from ase.visualize import view

        for idx, i in enumerate(convertedNumbers):
            E -= i * self.h
            neighbor_indices, _ = nl.get_neighbors(idx)
            # nlTmp = neighbor_indices.copy()
            # toDel = np.arange(27)
            # toDel = np.delete(toDel, nlTmp)
            # atomTmp = atoms.copy()
            # del atomTmp[toDel]
            # view(atomTmp)
            for j in convertedNumbers[neighbor_indices]:
                E -= self.J * i * j * 0.5

        self.results['energy'] = E
        print("Ising E =", E)

    # def has_energy(self, atomSymbol):
    #     # return True
    #     if atomSymbol in self.atom_energy_data:
    #         self.currAtomSymbol = atomSymbol
    #         return True
    #     else:
    #         return False