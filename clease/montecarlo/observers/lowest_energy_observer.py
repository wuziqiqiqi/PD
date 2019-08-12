from clease.montecarlo.observers import MCObserver
from clease import _logger
import numpy as np


class LowestEnergyStructure(MCObserver):
    """
    Observer that tracks the lowest energy state visited
    during an MC run

    atoms: Atoms
        Atoms object used in Monte Carlo

    verbose: bool
        If True progress messages will be printed
    """

    def __init__(self, atoms, verbose=True):
        self.atoms = atoms
        self.lowest_energy = np.inf
        self.lowest_energy_cf = None
        self.emin_atoms = None

        self.name = "LowestEnergyStructure"
        self.verbose = verbose

    def reset(self):
        self.lowest_energy = np.inf
        self.lowest_energy_cf = None
        self.emin_atoms = None

    def __call__(self, system_changes):
        """
        Checks if the current state has lower energy.
        If it has lower energy, the new state will be stored
        :param list system_changes: Last changes to the system
        """
        energy = self.atoms.get_calculator().results['energy']
        if (self.emin_atoms is None):
            calc = self.atoms.get_calculator()
            self.lowest_energy_cf = calc.get_cf()
            self.lowest_energy = energy
            self.emin_atoms = self.atoms.copy()
            return

        if energy < self.lowest_energy:
            dE = energy - self.lowest_energy
            calc = self.atoms.get_calculator()
            self.lowest_energy = energy
            self.emin_atoms = self.atoms.copy()
            self.lowest_energy_cf = calc.get_cf()
            if self.verbose:
                msg = "Found new low energy structure. "
                msg += "New energy: {} eV. ".format(self.lowest_energy)
                msg += "Change: {} eV".format(dE)
                _logger(msg)
