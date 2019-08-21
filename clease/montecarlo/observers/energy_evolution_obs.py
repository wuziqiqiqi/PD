from clease.montecarlo.observers import MCObserver
import numpy as np


class EnergyEvolution(MCObserver):
    """Trace the evolution of energy."""

    def __init__(self, calc):
        self.calc = calc
        self.energies = []
        MCObserver.__init__(self)
        self.name = "EnergyEvolution"

    def __call__(self, system_changes):
        """Append the current energy to the MC object.

        Parameters:

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        self.energies.append(self.calc.get_potential_energy())

    def reset(self):
        """Reset the history."""
        self.energies = []

    def save(self, fname="energy_evolution"):
        """Save the energy evolution in .csv file.

        Parameters:

        fname: str
            File name without the extension (.csv)
        """
        full_fname = fname + '.csv'
        np.savetxt(full_fname, self.energies, delimiter=",")
        print("Energy evolution data saved to {}".format(full_fname))
