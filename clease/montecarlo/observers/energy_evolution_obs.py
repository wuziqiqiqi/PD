from clease.montecarlo.observers import MCObserver
import numpy as np


class EnergyEvolution(MCObserver):
    """
    Class for tracing the evolution of energy
    """
    def __init__(self, calc):
        self.calc = calc
        self.energies = []
        MCObserver.__init__(self)
        self.name = "EnergyEvolution"

    def __call__(self, system_changes):
        """Append the current energy to the MC object."""
        self.energies.append(self.calc.get_potential_energy())

    def reset(self):
        """Reset the history."""
        self.energies = []

    def save(self, fname="energy_evolution.csv"):
        """Save the result to a numpy file."""
        np.savetxt(fname, self.energies, delimiter=",")
        print("Energy evolution data saved to {}".format(fname))
