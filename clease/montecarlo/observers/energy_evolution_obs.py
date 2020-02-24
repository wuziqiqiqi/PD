from clease.montecarlo.observers import MCObserver
import numpy as np


class EnergyEvolution(MCObserver):
    """Trace the evolution of energy."""

    def __init__(self, mc, ignore_reset=False):
        self.mc = mc
        self.energies = []
        self.mean_energies = []
        MCObserver.__init__(self)
        self.name = "EnergyEvolution"
        self.ignore_reset = ignore_reset

    def __call__(self, system_changes):
        """Append the current energy to the MC object.

        Parameters:

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        self.energies.append(self.mc.current_energy + self.mc.energy_bias)
        self.mean_energies.append(self.mc.mean_energy.mean +
                                  self.mc.energy_bias)

    def reset(self):
        """Reset the history."""
        if self.ignore_reset:
            return
        self.energies = []

    def save(self, fname: str = "energy_evolution") -> None:
        """Save the energy evolution in .csv file.

        :param fname: File name without the extension (.csv)
        """
        full_fname = fname + '.csv'
        np.savetxt(full_fname, self.energies, delimiter=",")
        print(f"Energy evolution data saved to {full_fname}")
