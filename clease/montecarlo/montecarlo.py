"""Monte Carlo method for ase."""
from typing import Dict, Union, Iterator, Any
from datetime import datetime
import time
import logging
import random
import math
from collections import Counter
from ase import Atoms
from ase.units import kB
from clease.datastructures import SystemChanges, MCStep
from .mc_evaluator import CEMCEvaluator, MCEvaluator
from .base import BaseMC
from .averager import Averager
from .bias_potential import BiasPotential
from .observers import MCObserver
from .trial_move_generator import TrialMoveGenerator, RandomSwap

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class Montecarlo(BaseMC):
    """Class for running Monte Carlo at a fixed composition (canonical).
    For more information, also see the documentation of the parent class
    :class:`~clease.montecarlo.base.BaseMC`.

    Args:
        system (Union[ase.Atoms, MCEvaluator]): Either an ASE Atoms object
            with an attached calculator, or a pre-initialized
            :class:`~clease.montecarlo.mc_evaluator.MCEvaluator`
            object.
        temp (float): Temperature of Monte Carlo simulation in Kelvin
        generator (TrialMoveGenerator, optional): A
            :class:`~clease.montecarlo.trial_move_generator.TrialMoveGenerator`
            object that produces trial moves. Defaults to None.
    """

    NAME = "MonteCarlo"

    def __init__(
        self,
        system: Union[Atoms, MCEvaluator],
        temp: float,
        generator: TrialMoveGenerator = None,
    ):
        # We cannot cause an energy calculation trigger in init,
        # so we defer these quantities until needed.
        self.current_energy = None
        self.new_energy = None
        self.mean_energy = None
        self.energy_squared = None

        super().__init__(system, temp)

        if generator is None:
            self.generator = _make_default_swap_generator(self.evaluator)
        else:
            self.generator = generator

        # List of observers that will be called every n-th step
        # similar to the ones used in the optimization routines
        self.observers = []
        self.bias_potentials = []

        self.current_step = 0
        self.num_accepted = 0
        self.status_every_sec = 30

        self.trial_move = []

        self.quit = False

    def _on_temp_change(self) -> None:
        """Reset the energy averagers after a change in temperature"""
        self.reset_averagers()
        self._reset_internal_counters()

    def update_current_energy(self) -> None:
        self.current_energy = self.evaluator.get_energy()
        self.current_energy += sum(
            bias.calculate_from_scratch(self.atoms) for bias in self.bias_potentials
        )
        logger.debug("Updating current energy to %s", self.current_energy)
        self.evaluator.reset()

    def _initialize_energies(self) -> None:
        """Initialize the energy quantities, causes a trigger of an energy evaluation."""

        self.update_current_energy()
        self.new_energy = self.current_energy

        # Create new averager objects, only the first time
        # (since they are initialized as None in __init__)
        if self.mean_energy is None:
            self.mean_energy = Averager(ref_value=self.current_energy)
        if self.energy_squared is None:
            self.energy_squared = Averager(ref_value=self.current_energy)

    def reset(self) -> None:
        """Reset all member variables to their original values."""
        logger.debug("Resetting.")
        for _, obs in self.observers:
            obs.reset()

        self.evaluator.reset()
        self._reset_internal_counters()
        self.reset_averagers()

    def reset_averagers(self) -> None:
        """Reset the energy averagers."""
        # Averagers are initialized to None in the constructor.
        for averager in (self.mean_energy, self.energy_squared):
            if averager is not None:
                averager.clear()

    def _reset_internal_counters(self) -> None:
        """Reset the step counters which are used internally"""
        self.current_step = 0
        self.num_accepted = 0

    def add_bias(self, potential: BiasPotential):
        """Add a new bias potential.

        Parameters:

        potential:
            Potential to be added
        """
        if not isinstance(potential, BiasPotential):
            raise TypeError("potential has to be of type BiasPotential")
        self.bias_potentials.append(potential)

    def attach(self, obs: MCObserver, interval: int = 1):
        """Attach observers to be called on a given MC step interval.

        Parameters:

        obs: MCObserver
            Observer to be added

        interval: int
            How often the observer should be called
        """
        if not obs.interval_ok(interval):
            name = type(obs).__name__
            raise ValueError(f"Invalid interval for {name}. Check docstring of the observer.")

        self.observers.append((interval, obs))

    def iter_observers(self) -> Iterator[MCObserver]:
        """Directly iterate the attached observers without also getting
        information about the interval."""
        # Remember that the observer list contains tuples of (interval, observer)
        for _, obs in self.observers:
            yield obs

    def initialize_run(self):
        """Prepare MC object for a new run."""
        logger.debug("Initializing run")
        self.generator.initialize(self.atoms)

        # Ensure the evaluator is properly synchronized.
        self.evaluator.synchronize()

        # Initialize/update relevant energy quantities
        self._initialize_energies()

        # Reset the internal step counters
        self._reset_internal_counters()

    def run(self, steps: int = 100, call_observers: bool = True) -> None:
        """Run Monte Carlo simulation.

        Parameters:

        steps: int
            Number of steps in the MC simulation
        call_observers: bool
            Should the observers be called during this run? Can be turned off for running burn-ins.
            The energy averagers will still be updated, even if this flag is disabled.
            Defaults to True.
        """

        # Construct the iterator, make the preparations for starting the run
        mc_iter = self.irun(steps, call_observers=call_observers)

        start = time.perf_counter()
        prev = self.current_step
        info_enabled = logger.isEnabledFor(logging.INFO)  # Do we emit INFO logs?
        for _ in mc_iter:
            # We only want to do this calculation if logging is enabled for INFO.
            if info_enabled and time.perf_counter() - start > self.status_every_sec:
                ms_per_step = 1000.0 * self.status_every_sec / (self.current_step - prev)
                logger.info(
                    "%d of %d steps. %.2f ms per step. Acceptance rate: %.2f",
                    self.current_step,
                    steps,
                    ms_per_step,
                    self.current_accept_rate,
                )
                prev = self.current_step
                start = time.perf_counter()

    def irun(self, steps: int, call_observers: bool = True) -> Iterator[MCStep]:
        """Run Monte Carlo simulation as an iterator.
        Can be used to inspect the MC after each step, for example,
        to print the energy every 5 steps, one could do:

        >>> mc = Montecarlo(...)  # doctest: +SKIP
        >>> for mc_step in mc.irun(500):  # doctest: +SKIP
        ...     if mc_step.step % 5 == 0:
        ...         print(f"Current energy: {mc_step.energy:.2f} eV")

        The iterator yields individual instances of :class:`~clease.datastructures.mc_step.MCStep`
        for each step which is taken.

        Parameters:

        steps: int
            Number of steps in the MC simulation
        call_observers: bool
            Should the observers be called during this run? Can be turned off for running burn-ins.
            The energy averagers will still be updated, even if this flag is disabled.
            Defaults to True.
        """
        logger.info("Starting MC run with %d steps.", steps)

        # We first ensure the MC object is initialized, and then we
        # construct the iterator.
        # This ensures everything is prepared before the first step
        # is consumed from the iterator.
        self.initialize_run()

        # Now we create the iterator
        return self._irun(steps, call_observers=call_observers)

    def _irun(self, steps: int, call_observers: bool = True) -> Iterator[MCStep]:
        """Create the MC iterator"""
        # Offset range by 1, so that we start with current step = 1
        for _ in range(steps):
            step = self._mc_step(call_observers=call_observers)

            en = step.energy
            self.mean_energy += en
            # E * E is slightly faster than E ** 2
            self.energy_squared += en * en

            if self.quit:
                # Breaking out will cause the iterator to end.
                logger.warning("Quit signal detected. Breaking the MC loop.")
                break
            # Yield, allow inspection before continuing.
            yield step
        else:
            # Loop was not broken, we reached the max number of steps
            logger.info("Reached maximum number of steps (%d mc steps)", steps)
        logger.debug("Reached end of MC iterator.")

    @property
    def meta_info(self) -> Dict[str, str]:
        """Return dict with meta info."""
        # Get the timestamp with millisecond precision
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        meta_info = {
            "timestamp": timestamp,
        }
        return meta_info

    @property
    def current_accept_rate(self) -> float:
        """Return the current accept rate as a value between 0 and 1."""
        if self.current_step == 0:
            # No steps have been taken yet.
            return 0.0
        return self.num_accepted / self.current_step

    def get_thermodynamic_quantities(self) -> Dict[str, Any]:
        """Compute thermodynamic quantities."""
        quantities = {}
        mean_energy = self.mean_energy.mean
        quantities["energy"] = mean_energy
        mean_sq = self.energy_squared.mean
        quantities["heat_capacity"] = (mean_sq - mean_energy**2) / (kB * self.temperature**2)
        quantities["energy_var"] = mean_sq - mean_energy**2
        quantities["temperature"] = self.temperature
        quantities["accept_rate"] = self.current_accept_rate
        quantities["n_mc_steps"] = self.current_step

        at_count = self.count_atoms()
        for key, value in at_count.items():
            name = f"{key}_conc"
            conc = value / len(self.atoms)
            quantities[name] = conc

        # Add some more info that can be useful
        quantities.update(self.meta_info)

        # Add information from observers
        quantities.update(self._get_obs_averages())
        return quantities

    def _get_obs_averages(self) -> Dict[str, Any]:
        """Get average measurements from observers"""
        obs_avgs = {}
        for obs in self.iter_observers():
            obs_avgs.update(obs.get_averages())
        return obs_avgs

    def _calculate_step(self, system_changes: SystemChanges):
        """Calculate energies given a step, and decide if we accept the step.

        Returns boolean if system changes are accepted.

        Parameters:

        system_changes: list
            List with system changes
        """
        self.evaluator.apply_system_changes(system_changes)
        # Evaluate the energy quantity after applying the changes to the system.
        self.new_energy = self.evaluator.get_energy(applied_changes=system_changes)

        # NOTE: As this is called after calculate, the changes has
        # already been introduced to the system
        for bias in self.bias_potentials:
            self.new_energy += bias(system_changes)
        accept = self._do_accept(self.current_energy, self.new_energy)

        if accept:
            # Changes accepted, finalize evaluator.
            self.evaluator.keep_system_changes(system_changes)
        else:
            # Undo changes
            self.evaluator.undo_system_changes(system_changes)

        return accept

    def _do_accept(self, current_energy: float, new_energy: float) -> bool:
        """Decide if we accept a state, based on the energies.

        Return a bool on whether the move was accepted.

        Parameters:

        :param current_energy: Energy of the current configuration
        :param new_energy: Energy of the new configuration.
        """
        # Standard Metropolis acceptance criteria
        if new_energy < current_energy:
            return True
        energy_diff = new_energy - current_energy
        probability = math.exp(-energy_diff / self.kT)

        return random.random() <= probability

    def _move_accepted(self, system_changes: SystemChanges) -> None:
        self.num_accepted += 1
        self.generator.on_move_accepted(system_changes)
        self.current_energy = self.new_energy

    def _move_rejected(self, system_changes: SystemChanges) -> None:
        self.generator.on_move_rejected(system_changes)

    def count_atoms(self) -> Dict[str, int]:
        """Count the number of each element."""
        return dict(Counter(self.atoms.symbols))

    def _mc_step(self, call_observers: bool = True) -> MCStep:
        """Make one Monte Carlo step by swithing two atoms."""
        self.current_step += 1
        system_changes = self.generator.get_trial_move()
        self.trial_move = system_changes

        # Calculate step, and whether we accept it
        move_accepted = self._calculate_step(system_changes)

        updater = self._move_accepted if move_accepted else self._move_rejected
        updater(system_changes)

        step = MCStep(self.current_step, self.current_energy, move_accepted, system_changes)
        if call_observers:
            # Execute all observers
            self.execute_observers(step)

        return step

    def execute_observers(self, last_step: MCStep):
        for interval, obs in self.observers:
            if self.current_step % interval == 0:
                obs.observe_step(last_step)


def _make_default_swap_generator(evaluator: MCEvaluator) -> RandomSwap:
    """Construct the default RandomSwap trial move generator.
    If the evaluator object is not a Cluster Expansion evaluator,
    then any swaps are allowed.

    If the evaluator is a Cluster Expansion MC evaluator object,
    then only non-background sites are considered for random swaps.
    This does *not* constrain swaps between sublattices.
    """
    atoms = evaluator.atoms
    if not isinstance(evaluator, CEMCEvaluator):
        # Running a non-CE Evaluator
        return RandomSwap(atoms)
    # We're a CE evaluator, constrain the background sites
    non_bkg_indices = evaluator.settings.non_background_indices
    return RandomSwap(atoms, indices=non_bkg_indices)
