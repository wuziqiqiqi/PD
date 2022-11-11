import numpy as np
import attr
from clease.calculator import Clease
from clease.montecarlo.averager import Averager
from clease.datastructures.mc_step import MCStep
from .mc_observer import MCObserver


@attr.define(slots=True)
class SGCQuantities:
    """Container for observed quantities in the SGCObserver"""

    energy: Averager = attr.field()
    energy_sq: Averager = attr.field()
    singlets: np.ndarray = attr.field()
    singlets_sq: np.ndarray = attr.field()
    singl_eng: np.ndarray = attr.field()
    counter: int = attr.field(default=0)

    def reset(self) -> None:
        self.counter = 0
        self.energy.clear()
        self.energy_sq.clear()
        self.singlets[:] = 0.0
        self.singlets_sq[:] = 0.0
        self.singl_eng[:] = 0.0


class SGCObserver(MCObserver):
    """
    Observer mainly intended to track additional quantities needed when
    running SGC Monte Carlo. This observer has to be executed on every MC step.

    Parameters:

    calc: `clease.calculators.Clease`
        Clease calculator

    observe_singlets: bool
        Whether the singlet values of the calculator are measured during each observation.
        Measuring singlets is slightly more expensive, so this is disabled by default.
    """

    name = "SGCObersver"

    def __init__(self, calc: Clease, observe_singlets: bool = False):
        self.observe_singlets = observe_singlets
        super().__init__()
        self.calc = calc
        initial_energy = calc.get_potential_energy()
        n_singlets = len(self.calc.get_singlets())
        self.quantities = SGCQuantities(
            energy=Averager(ref_value=initial_energy),
            energy_sq=Averager(ref_value=initial_energy**2),
            singlets=np.zeros(n_singlets, dtype=np.float64),
            singlets_sq=np.zeros(n_singlets, dtype=np.float64),
            singl_eng=np.zeros(n_singlets, dtype=np.float64),
        )

    def reset(self):
        """Reset all variables to zero."""
        self.quantities.reset()

    def get_current_energy(self) -> float:
        """Return the current energy of the attached calculator object."""
        return self.calc.results["energy"]

    def observe_step(self, mc_step: MCStep):
        """Update all SGC parameters."""
        # Reference to avoid self. lookup multiple times
        quantities = self.quantities
        E = self.get_current_energy()

        quantities.counter += 1
        quantities.energy += E
        quantities.energy_sq += E * E

        if self.observe_singlets:
            # Only perform the singlet observations if requested.
            new_singlets = self.calc.get_singlets()
            quantities.singlets += new_singlets
            quantities.singlets_sq += new_singlets * new_singlets
            quantities.singl_eng += new_singlets * E

    @property
    def energy(self):
        return self.quantities.energy

    @property
    def energy_sq(self):
        return self.quantities.energy_sq

    @property
    def singlets(self):
        return self.quantities.singlets

    @property
    def singl_eng(self):
        return self.quantities.singl_eng

    @property
    def counter(self):
        return self.quantities.counter

    def interval_ok(self, interval):
        return interval == 1
