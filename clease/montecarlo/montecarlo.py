# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
import sys
import datetime
import time
import logging
from typing import Sequence, Tuple
import numpy as np
from numpy.random import choice
from ase import Atoms
from ase.units import kB
from clease.tools import SystemChange
from clease.montecarlo.exponential_filter import ExponentialFilter
from clease.montecarlo.averager import Averager
from clease.montecarlo import BiasPotential
from clease.montecarlo.observers import MCObserver
from clease.montecarlo.swap_move_index_tracker import SwapMoveIndexTracker

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

    Parameters:

    atoms: Atoms object
        ASE Atoms object (with CE calculator attached)

    temp: float
        Temperature of Monte Carlo simulation in Kelvin
    """

    def __init__(self, atoms: Atoms, temp: float):
        self.name = "MonteCarlo"
        self.atoms = atoms
        self.active_indices = list(range(len(atoms)))
        self.T = temp

        # List of observers that will be called every n-th step
        # similar to the ones used in the optimization routines
        self.observers = []

        self.constraints = []
        self.max_allowed_constraint_pass_attempts = 1000

        if self.max_allowed_constraint_pass_attempts <= 0:
            raise ValueError("Max. constraint attempts has to be > 0!")

        self.bias_potentials = []

        self.current_step = 0
        self.num_accepted = 0
        self.status_every_sec = 30
        self.atoms_tracker = SwapMoveIndexTracker()
        self.symbols = []
        E0 = self.atoms.calc.calculate(None, None, None)
        self.atoms.calc.clear_history()
        self._build_atoms_list()
        self.current_energy = E0
        self.new_energy = self.current_energy

        self.trial_move = []  # Last trial move performed
        self.mean_energy = Averager(ref_value=E0)
        self.energy_squared = Averager(ref_value=E0)

        # Some member variables used to update the atom tracker, only relevant
        # for canonical MC
        self.rand_a = 0
        self.rand_b = 0
        self.selected_a = 0
        self.selected_b = 0
        self.quit = False

        self.filter = ExponentialFilter(min_time=0.2 * len(self.atoms),
                                        max_time=20 * len(self.atoms),
                                        n_subfilters=10)

    def insert_symbol(self, symb: str, indices: Sequence[int]):
        """Insert symbols on a predefined set of indices.

        Parameters:

        symb: str
            Symbol to be inserted
        indices: list
            Indices where symb should be inserted
        """
        calc = self.atoms.calc
        for indx in indices:
            system_changes = SystemChange(index=indx,
                                          old_symb=self.atoms[indx].symbol,
                                          new_symb=symb)

            if not self._no_constraint_violations([system_changes]):
                msg = ("The indices given results in an update "
                       "that violate one or more constraints!")
                logger.error(msg)
                raise ValueError(msg)
            calc.update_cf(system_changes)
        self._build_atoms_list()
        calc.clear_history()

    def insert_symbol_random_places(self, symbol: str, num: int = 1, swap_symbs: Tuple[str] = ()):
        """Insert random symbol.

        Parameters:

        symbol:
            Symbol to insert

        num:
            Number of sites to insert

        swap_symbs:
            If given, will insert the replace symbol with sites having symbols
            in this list
        """
        if not swap_symbs:
            symbs = [atom.symbol for atom in self.atoms]
            swap_symbs = list(set(symbs))
        num_inserted = 0
        max_attempts = 10 * len(self.atoms)

        calc = self.atoms.calc
        attempts = 0
        while num_inserted < num and attempts < max_attempts:
            attempts += 1
            old_symb = choice(swap_symbs)
            if old_symb == symbol:
                continue

            indx = self.atoms_tracker.get_random_indx_of_symbol(old_symb)
            if self.atoms[indx].symbol not in swap_symbs:
                # This can happen because the atom_indx is inconsistent
                # until after all the atoms have been inserted
                continue

            system_changes = SystemChange(index=indx, old_symb=old_symb, new_symb=symbol)
            if not self._no_constraint_violations([system_changes]):
                continue
            calc.update_cf(system_changes)
            num_inserted += 1
        self._build_atoms_list()
        calc.clear_history()

        if attempts == max_attempts:
            raise RuntimeError(f"Could insert {num} {symbol} atoms!")

    def update_current_energy(self):
        """Enforce a new energy evaluation."""
        self.current_energy = self.atoms.get_potential_energy()

    def set_symbols(self, symbs: Sequence[str]):
        """Set the symbols of this Monte Carlo run.

        Parameters:

        symbs:
            Symbols to insert. Has to have the same length as the
            attached atoms object.
        """
        self.atoms.calc.set_symbols(symbs)
        self._build_atoms_list()
        self.update_current_energy()

    def _check_symbols(self):
        """Check that there is at least two different symbols."""
        symbs = [atom.symbol for atom in self.atoms]
        count = {}
        for symb in symbs:
            if symb not in count:
                count[symb] = 1
            else:
                count[symb] += 1

        # Verify that there is at two elements with more that two symbols
        if len(count.keys()) < 2:
            raise TooFewElementsError("There is only one element in the given atoms object!")
        n_elems_more_than_2 = 0
        for _, value in count.items():
            if value >= 2:
                n_elems_more_than_2 += 1
        if n_elems_more_than_2 < 2:
            raise TooFewElementsError("There is only one element that has more than one atom")

    def _no_constraint_violations(self, system_changes: Sequence[SystemChange]):
        """Check if the proposed moves violate any of the constraints.

        Parameters:

        system_changes: list
            Changes of the proposed move
        """
        logger.debug('Checking system change: %s', system_changes)
        for constraint in self.constraints:
            if not constraint(system_changes):
                logger.debug('System change rejected by constraint %s', constraint.name)
                return False
        logger.debug('System change does not violate constraints.')
        return True

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

    def _build_atoms_list(self):
        """
        Create a dictionary of the indices of each atom which is used to
        make sure that two equal atoms cannot be swapped
        """
        self.atoms_tracker.init_tracker(self.atoms)
        self.symbols = self.atoms_tracker.symbols

    def _update_tracker(self, system_changes: Sequence[SystemChange]):
        """Update the atom tracker."""
        self.atoms_tracker.update_swap_move(system_changes)

    def add_constraint(self, constraint):
        """Add a new constraint to the sampler.

        Parameters:

        constraint: MCConstraint
            Instance of a constraint object
        """
        self.constraints.append(constraint)

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

    def run(self, steps: int = 100):
        """Run Monte Carlo simulation.

        Parameters:

        steps: int
            Number of steps in the MC simulation
        """
        # Check the number of different elements are correct to avoid
        # infinite loops
        self._check_symbols()

        self.update_current_energy()

        # Atoms object should have attached calculator
        # Add check that this is show
        self._mc_step()

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

    def _get_trial_move(self):
        """
        Perform a trial move by swapping two atoms

        """
        symb_a = choice(self.symbols)
        symb_b = choice([s for s in self.symbols if s != symb_a])
        rand_pos_a = self.atoms_tracker.get_random_indx_of_symbol(symb_a)
        rand_pos_b = self.atoms_tracker.get_random_indx_of_symbol(symb_b)
        system_changes = [
            SystemChange(index=rand_pos_a, old_symb=symb_a, new_symb=symb_b),
            SystemChange(index=rand_pos_b, old_symb=symb_b, new_symb=symb_a)
        ]

        logger.debug('Proposed system changes: %s', system_changes)
        return system_changes

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
        self._update_tracker(system_changes)
        self.current_energy = self.new_energy
        return system_changes

    # pylint: disable=no-self-use
    def _move_rejected(self, system_changes: Sequence[SystemChange]) -> Sequence[SystemChange]:
        logger.debug('Move rejected, undoing system changes: %s', system_changes)
        # Move rejected, no changes are made
        system_changes = [
            SystemChange(index=change.index, old_symb=change.old_symb, new_symb=change.old_symb)
            for change in system_changes
        ]
        logger.debug('Reversed system changes: %s', system_changes)
        return system_changes

    def count_atoms(self):
        """Count the number of each element."""
        atom_count = {key: 0 for key in self.symbols}
        for atom in self.atoms:
            atom_count[atom.symbol] += 1
        return atom_count

    def _mc_step(self):
        """Make one Monte Carlo step by swithing two atoms."""
        self.current_step += 1

        system_changes = self._get_trial_move()
        counter = 0
        while not self._no_constraint_violations(system_changes) and \
                counter < self.max_allowed_constraint_pass_attempts:
            system_changes = self._get_trial_move()
            counter += 1

        self.trial_move = system_changes

        if counter == self.max_allowed_constraint_pass_attempts:
            msg = "Did not manage to produce a trial move that does not "
            msg += "violate any of the constraints"
            logger.error(msg)
            raise CanNotFindLegalMoveError(msg)

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
