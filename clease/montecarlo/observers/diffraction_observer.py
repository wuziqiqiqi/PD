from clease.montecarlo.observers import MCObserver
import numpy as np


class DiffractionUpdater(object):
    """
    Utility class for all objects that require tracing of a fourier
    reflection.

    Parameters:
    atoms: Atoms
        Atoms object used in Monte Carlo

    k_vector: list
        Fourier reflection to be traced

    list active_symbols: list
        List of symbols that reflects

    list all_symbols: list
        List of all symbols in the simulation
    """
    def __init__(self, atoms=None, k_vector=[], active_symbols=[], 
                 all_symbols=[]):
        MCObserver.__init__(self)
        self.orig_symbols = [atom.symbol for atom in atoms]
        self.k_vector = k_vector
        self.N = len(atoms)
        self.k_dot_r = atoms.get_positions().dot(self.k_vector)
        self.indicator = {k: 0 for k in all_symbols}
        for symb in active_symbols:
            self.indicator[symb] = 1.0

        self.value = self.calculate_from_scratch(self.orig_symbols)
        self.prev_value = self.value

    def update(self, system_changes):
        """
        Update the reflection value
        """
        self.prev_value = self.value
        for change in system_changes:
            f_val = np.exp(1j*self.k_dot_r[change[0]])/self.N
            self.value += self.indicator[change[2]]*f_val
            self.value -= self.indicator[change[1]]*f_val

    def undo(self):
        """
        Undo the last update
        """
        self.value = self.prev_value

    def reset(self):
        """
        Reset all values
        """
        self.value = self.calculate_from_scratch(self.orig_symbols)
        self.prev_value = self.value

    def calculate_from_scratch(self, symbols):
        """Calculate the intensity from sctrach."""
        value = 0.0 + 1j*0.0
        for i, symb in enumerate(symbols):
            value += self.indicator[symb]*np.exp(1j*self.k_dot_r[i])
        return value / len(symbols)


class DiffractionObserver(MCObserver):
    """
    Observer that traces the reflection intensity
    See docstring of DiffractionUpdater for explination of the arguments.
    """
    def __init__(self, atoms=None, k_vector=[], active_symbols=[],
                 all_symbols=[], name="reflect"):
        MCObserver.__init__(self)
        self.updater = DiffractionUpdater(
            atoms=atoms, k_vector=k_vector, active_symbols=active_symbols,
            all_symbols=all_symbols)
        self.avg = self.updater.value
        self.num_updates = 1
        self.name = name

    def __call__(self, system_changes):
        self.updater.update(system_changes)
        self.avg += self.updater.value

    def get_averages(self):
        return {self.name: np.abs(self.avg/self.num_updates)}

    def reset(self):
        self.updater.reset()
        self.avg = self.updater.value
        self.num_updates = 1
