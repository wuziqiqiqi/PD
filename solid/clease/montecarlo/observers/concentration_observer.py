from typing import Dict
import ase
from clease.datastructures import MCStep, SystemChanges
from clease.montecarlo.averager import Averager
from .mc_observer import MCObserver


class ConcentrationObserver(MCObserver):
    """
    Observer that can be attached to a MC run, to track the concenctration of a
    particular element. This observer has to be executed on every MC step.

    Parameters:

    atoms: Atoms object
        Atoms object used for MC

    element: str
        The element that should be tracked
    """

    name = "ConcentrationObserver"

    def __init__(self, atoms: ase.Atoms, element: str):
        super().__init__()
        self.element = element
        self.n = len(atoms)
        self.init_conc = self.calculate_from_scratch(atoms)
        self.current_conc = self.init_conc
        self._make_averagers()

    def new_concentration(self, system_changes: SystemChanges) -> float:
        """Calculate the new consentration after the changes."""
        new_conc = self.current_conc
        for change in system_changes:
            if change.new_symb == self.element:
                new_conc += 1.0 / self.n
            if change.old_symb == self.element:
                new_conc -= 1.0 / self.n
        return new_conc

    def __call__(self, system_changes: SystemChanges, peak: bool = False) -> float:
        """Implement the __call__ method to work with the BiasPotentials"""
        if not system_changes:
            # No changes, None or some other falsey thing.
            return self.current_conc

        new_conc = self.new_concentration(system_changes)
        if peak:
            return new_conc

        self.current_conc = new_conc
        self.avg_conc += new_conc
        self.avg_conc_sq += new_conc**2
        return self.current_conc

    def observe_step(self, mc_step: MCStep, peak: bool = False) -> float:
        if mc_step.move_rejected:
            return self.current_conc

        return self(mc_step.last_move, peak=peak)

    def reset(self) -> None:
        """Reset the observer"""
        # Remake new average objects, since we also change the reference concentration
        self._make_averagers()

    def _make_averagers(self) -> None:
        """Construct the internal averager objects. Starts with the current concentration
        as the first sample."""
        self.avg_conc = Averager(ref_value=self.current_conc)
        self.avg_conc_sq = Averager(ref_value=self.current_conc**2)
        # The current concentration is the first sample
        self.avg_conc += self.current_conc
        self.avg_conc_sq += self.current_conc**2

    def get_averages(self) -> Dict[str, float]:
        mean_conc = self.avg_conc.mean
        var_conc = self.avg_conc_sq.mean - mean_conc**2
        return {f"conc_{self.element}": mean_conc, f"conc_var_{self.element}": var_conc}

    def calculate_from_scratch(self, atoms: ase.Atoms) -> float:
        """Calculate the concentration of the element in the atoms object."""
        num_atoms = sum(atoms.symbols == self.element)
        return num_atoms / len(atoms)

    def interval_ok(self, interval: int) -> bool:
        """Every step must be observed, as otherwise we'd miss updates,
        and the concentration becomes incorerct."""
        return interval == 1
