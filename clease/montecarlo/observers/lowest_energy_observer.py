import logging
import numpy as np
import ase
from clease.datastructures import MCStep
from clease.calculator import CleaseCacheCalculator
from .mc_observer import MCObserver

logger = logging.getLogger(__name__)


class LowestEnergyStructure(MCObserver):
    """Track the lowest energy state visited during an MC run.

    atoms: Atoms object
        Atoms object used in Monte Carlo

    track_cf: bool
        Whether to keep a copy of the correlation functions for the
        emin structure. If enabled, this will be stored in the ``lowest_energy_cf``
        variable. Defaults to False.

    verbose: bool
        If `True`, progress messages will be printed
    """

    name = "LowestEnergyStructure"

    def __init__(self, atoms: ase.Atoms, track_cf: bool = False, verbose: bool = False):
        super().__init__()
        self.atoms = atoms
        self.track_cf = track_cf
        self.verbose = verbose
        # Silence pylint
        self.lowest_energy_cf = None
        self.reset()

    def reset(self) -> None:
        self.lowest_energy_cf = None
        self.emin_atoms: ase.Atoms = self.atoms.copy()
        # Keep a reference directly to the calculator cache
        self._calc_cache = CleaseCacheCalculator()
        self.emin_atoms.calc = self._calc_cache
        self.lowest_energy = np.inf

    @property
    def lowest_energy(self) -> float:
        return self.emin_results["energy"]

    @lowest_energy.setter
    def lowest_energy(self, value: float) -> None:
        """The currently best observed energy"""
        self.emin_results["energy"] = value
        # Keep a copy without needing to look up the calculator
        # all the time, for quicker lookup.
        self._lowest_energy_cache = value

    @property
    def emin_results(self) -> dict:
        """The results dictionary of the lowest energy atoms"""
        return self._calc_cache.results

    @property
    def calc(self):
        return self.atoms.calc

    @property
    def energy(self):
        """The energy of the current atoms object (not the emin energy)"""
        return self.calc.results["energy"]

    def observe_step(self, mc_step: MCStep) -> None:
        """
        Check if the current state has lower energy and store the current
        state if it has a lower energy than the previous state.

        mc_step: MCStep
             Instance of MCStep with information on the latest step.
        """
        # We do the comparison even if the move was rejected,
        # as no energy has been recorded for the first step, yet.
        # The internal energy cache is np.inf when nothing has been recorded.
        if mc_step.energy < self._lowest_energy_cache:
            self._update_emin(mc_step)
            dE = mc_step.energy - self.lowest_energy
            msg = "Found new low energy structure. New energy: %.3f eV. Change: %.3f eV"
            logger.info(msg, self.lowest_energy, dE)
            if self.verbose:
                print(msg % (self.lowest_energy, dE))

    def _update_emin(self, mc_step: MCStep) -> None:
        if self.track_cf:
            self.lowest_energy_cf = self.calc.get_cf()
        self.lowest_energy = mc_step.energy

        # Avoid copying the atoms object multiple times. We can only ever have
        # changed the symbols, so copy those over from the atoms object.
        self.emin_atoms.numbers = self.atoms.numbers
