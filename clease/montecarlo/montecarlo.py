# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
import sys
import datetime
import time
import logging
from collections import Counter
from typing import Sequence
import numpy as np
from ase import Atoms
from ase.units import kB
from clease.tools import SystemChange
from clease.montecarlo.exponential_filter import ExponentialFilter
from clease.montecarlo.averager import Averager
from clease.montecarlo.bias_potential import BiasPotential
from clease.montecarlo.observers import MCObserver
from clease.montecarlo.trial_move_generator import TrialMoveGenerator, RandomSwap

logger = logging.getLogger(__name__)


class DidNotReachEquillibriumError(Exception):
    pass


class TooFewElementsError(Exception):
    pass


class CanNotFindLegalMoveError(Exception):
    pass


# pylint: disable=too-many-instance-attributes
class Montecarlo:
    """Class for running Monte Carlo at a fixed composition (canonical).

    :param atoms: ASE Atoms object (with CE calculator attached)
    :param temp: Temperature of Monte Carlo simulation in Kelvin
    :param generator: Generator that produces trial moves
    """

    def __init__(self, atoms: Atoms, temp: float, generator: TrialMoveGenerator = None):
        self.name = "MonteCarlo"
        self.atoms = atoms
        self.T = temp

        if generator is None:
            self.generator = RandomSwap(atoms)
        else:
            self.generator = generator

        # List of observers that will be called every n-th step
        # similar to the ones used in the optimization routines
        self.observers = []
        self.bias_potentials = []

        self.current_step = 0
        self.num_accepted = 0
        self.status_every_sec = 30
        E0 = self.atoms.calc.calculate(None, None, None)
        self.atoms.calc.clear_history()
        self.current_energy = E0
        self.new_energy = self.current_energy

        self.trial_move = []  # Last trial move performed
        self.mean_energy = Averager(ref_value=self.current_energy)
        self.energy_squared = Averager(ref_value=self.current_energy)

        # Some member variables used to update the atom tracker, only relevant
        # for canonical MC
        self.quit = False

        self.filter = ExponentialFilter(min_time=0.2 * len(self.atoms),
                                        max_time=20 * len(self.atoms),
                                        n_subfilters=10)

    def update_current_energy(self):
        """Enforce a new energy evaluation."""
        self.current_energy = self.atoms.get_potential_energy()

    def _check_symbols(self):
        """Check that there is at least two different symbols."""
        count = self.count_atoms()
        # Verify that there is at two elements with more that two symbols
        if len(count) < 2:
            raise TooFewElementsError("There is only one element in the given atoms object!")
        n_elems_more_than_2 = 0
        for _, value in count.items():
            if value >= 2:
                n_elems_more_than_2 += 1
        if n_elems_more_than_2 < 2:
            raise TooFewElementsError("There is only one element that has more than one atom")

    def reset(self):
        """Reset all member variables to their original values."""
        logger.debug('Resetting.')
        for _, obs in self.observers:
            obs.reset()

        self.filter.reset()
        self.current_step = 0
        self.num_accepted = 0
        self.mean_energy.clear()
        self.energy_squared.clear()

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

    def _initialize_run(self):
        self._check_symbols()
        self.generator.initialize(self.atoms)

        self.update_current_energy()

        # Atoms object should have attached calculator
        # Add check that this is show
        self._mc_step()

    def run(self, steps: int = 100):
        """Run Monte Carlo simulation.

        Parameters:

        steps: int
            Number of steps in the MC simulation
        """
        # Check the number of different elements are correct to avoid
        # infinite loops
        self._initialize_run()

        start = time.time()
        prev = self.current_step
        while self.current_step < steps:
            E, _ = self._mc_step()

            self.mean_energy += E
            self.energy_squared += E**2

            if time.time() - start > self.status_every_sec:
                ms_per_step = 1000.0 * self.status_every_sec / (self.current_step - prev)
                accept_rate = self.num_accepted / self.current_step
                logger.info("%d of %d steps. %.2f ms per step. Acceptance rate: %.2f",
                            self.current_step, steps, ms_per_step, accept_rate)
                prev = self.current_step
                start = time.time()

            if self.quit:
                logger.debug('Quit signal detected. Breaking.')
                break

        logger.info("Reached maximum number of steps (%d mc steps)", steps)

    @property
    def meta_info(self):
        """Return dict with meta info."""
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        v_info = sys.version_info
        meta_info = {
            "timestamp": st,
            "python_version": f"{v_info.major}.{v_info.minor}.{v_info.micro}"
        }
        return meta_info

    def get_thermodynamic_quantities(self):
        """Compute thermodynamic quantities."""
        quantities = {}
        mean_energy = self.mean_energy.mean
        quantities["energy"] = mean_energy
        mean_sq = self.energy_squared.mean
        quantities["heat_capacity"] = \
            (mean_sq - mean_energy**2) / (kB * self.T**2)
        quantities["energy_var"] = mean_sq - mean_energy**2
        quantities["temperature"] = self.T
        at_count = self.count_atoms()
        for key, value in at_count.items():
            name = f"{key}_conc"
            conc = float(value) / len(self.atoms)
            quantities[name] = conc

        # Add some more info that can be useful
        quantities.update(self.meta_info)

        # Add information from observers
        for obs in self.observers:
            quantities.update(obs[1].get_averages())
        return quantities

    def _calculate_step(self, system_changes: Sequence[SystemChange]):
        """Calculate energies given a step, and decide if we accept the step.

        Returns boolean if system changes are accepted.

        Parameters:

        system_changes: list
            List with system changes
        """
        # NOTE: Calculate updates the system
        calc = self.atoms.calc

        with calc.with_system_changes(system_changes) as keeper:
            self.new_energy = calc.get_energy()

            # NOTE: As this is called after calculate, the changes has
            # already been introduced to the system
            for bias in self.bias_potentials:
                self.new_energy += bias(system_changes)

            logger.debug('Current energy: %.3f eV, new energy: %.3f eV', self.current_energy,
                         self.new_energy)
            accept = self._do_accept(self.current_energy, self.new_energy)
            logger.debug('Change accepted? %s', accept)

            # Decide if we keep changes, or rollback
            keeper.keep_changes = accept
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
        kT = self.T * kB
        energy_diff = new_energy - current_energy
        probability = np.exp(-energy_diff / kT)
        logger.debug('Calculated probability: %.3f', probability)
        return np.random.rand() <= probability

    def _move_accepted(self, system_changes: Sequence[SystemChange]) -> Sequence[SystemChange]:
        logger.debug('Move accepted, updating things')
        self.num_accepted += 1
        self.generator.on_move_accepted(system_changes)
        self.current_energy = self.new_energy
        return system_changes

    # pylint: disable=no-self-use
    def _move_rejected(self, system_changes: Sequence[SystemChange]) -> Sequence[SystemChange]:
        logger.debug('Move rejected, undoing system changes: %s', system_changes)
        self.generator.on_move_rejected(system_changes)

        # Move rejected, no changes are made
        system_changes = [
            SystemChange(index=change.index,
                         old_symb=change.old_symb,
                         new_symb=change.old_symb,
                         name=change.name) for change in system_changes
        ]
        logger.debug('Reversed system changes: %s', system_changes)
        return system_changes

    def count_atoms(self):
        """Count the number of each element."""
        return dict(Counter(self.atoms.symbols))

    def _mc_step(self):
        """Make one Monte Carlo step by swithing two atoms."""
        self.current_step += 1

        system_changes = self.generator.get_trial_move()
        self.trial_move = system_changes

        # Calculate step, and whether we accept it
        move_accepted = self._calculate_step(system_changes)

        updater = self._move_accepted if move_accepted else self._move_rejected
        system_changes = updater(system_changes)

        # Execute all observers
        self.execute_observers(system_changes)
        self.filter.add(self.current_energy)
        return self.current_energy, move_accepted

    def execute_observers(self, system_changes: Sequence[SystemChange]):
        for interval, obs in self.observers:
            if self.current_step % interval == 0:
                logger.debug('Executing observer %s at step %d with interval %d.', obs.name,
                             self.current_step, interval)
                obs(system_changes)
