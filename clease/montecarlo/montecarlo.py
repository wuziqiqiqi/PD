# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
from __future__ import division
import time
import numpy as np
from ase.units import kB
from clease.montecarlo.exponential_filter import ExponentialFilter
from clease.montecarlo.averager import Averager
from clease.montecarlo import BiasPotential
from clease.montecarlo .swap_move_index_tracker import SwapMoveIndexTracker
from clease import _logger


class DidNotReachEquillibriumError(Exception):
    pass


class TooFewElementsError(Exception):
    pass


class CanNotFindLegalMoveError(Exception):
    pass


class Montecarlo(object):
    """Class for running Monte Carlo at a fixed composition (canonical).

    Parameters:

    atoms: Atoms object
        ASE Atoms object (with CE calculator attached)

    temp: float
        Temperature of Monte Carlo simulation in Kelvin
    """

    def __init__(self, atoms, temp):
        self.name = "MonteCarlo"
        self.atoms = atoms
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
        E0 = self.atoms.get_calculator().calculate(None, None, None)
        self.atoms.get_calculator().clear_history()
        self._build_atoms_list()
        self.current_energy = E0
        self.bias_energy = 0.0
        self.new_bias_energy = self.bias_energy
        self.new_energy = self.current_energy

        # Keep the energy of old and trial state
        self.last_energies = np.zeros(2)
        self.trial_move = []  # Last trial move performed
        self.mean_energy = Averager(ref_value=E0)
        self.energy_squared = Averager(ref_value=E0)
        self.energy_bias = 0.0
        self.update_energy_bias = True

        # Some member variables used to update the atom tracker, only relevant
        # for canonical MC
        self.rand_a = 0
        self.rand_b = 0
        self.selected_a = 0
        self.selected_b = 0
        self.quit = False

        self.filter = ExponentialFilter(min_time=0.2*len(self.atoms),
                                        max_time=20*len(self.atoms),
                                        n_subfilters=10)

    def _probe_energy_bias(self, num_sweeps=2):
        """
        Run MC steps to probe the energy bias. The bias
        will be subtracted off the zeroth ECI and then
        added to the total energy during post processing.
        """
        self.energy_bias = 0.0
        num_steps = num_sweeps*len(self.atoms)
        self.log("Probing energy bias using {} MC steps...".format(num_steps))
        for _ in range(num_steps):
            E, _ = self._mc_step()
            self.mean_energy += E
            self.energy_squared += E**2

        self.log("Energy after probing: {}".format(self.current_energy))
        self.energy_bias = self.current_energy
        self._remove_bias_from_empty_eci(self.energy_bias)

    def _remove_bias_from_empty_eci(self, bias):
        """Remove the energy bias from the empty cluster's ECI.

        Parameters:

        bias: float
            Energy bias
        """
        calc = self.atoms.get_calculator()
        eci = calc.eci
        c0_eci = eci['c0']
        c0_eci -= bias/len(self.atoms)
        eci['c0'] = c0_eci

        calc.update_eci(eci)

        # Force re-calculation of the energy
        self.current_energy = calc.calculate(None, None, None)
        self.last_energies[0] = self.current_energy

        if abs(self.current_energy) > 1E-6:
            raise RuntimeError("Energy is not 0 after subtracting "
                               "the bias. Got {}".format(self.current_energy))

        self.log('Bias subtracted from empty cluster...')

    def _undo_energy_bias_from_eci(self):
        eci = self.atoms.get_calculator().eci
        eci['c0'] += self.energy_bias/len(self.atoms)
        self.atoms.get_calculator().update_eci(eci)
        calc = self.atoms.get_calculator()
        self.current_energy = calc.calculate(None, None, None)
        #self.energy_bias = 0.0
        self.log('Empty cluster ECI reset to original value...')

    def insert_symbol(self, symb, indices):
        """Insert symbols on a predefined set of indices.

        Parameters:

        symb: str
            Symbol to be inserted
        indices: list
            Indices where symb should be inserted
        """
        calc = self.atoms.get_calculator()
        for indx in indices:
            system_changes = (indx, self.atoms[indx].symbol, symb)

            if not self._no_constraint_violations([system_changes]):
                raise ValueError("The indices given results in an update "
                                 "that violate one or more constraints!")
            calc.update_cf(system_changes)
        self._build_atoms_list()
        calc.clear_history()

    def insert_symbol_random_places(self, symbol, num=1, swap_symbs=[]):
        """Insert random symbol.

        Parameters:

        symbol: str
            Symbol to insert

        num: int
            Number of sites to insert

        swap_symbs: list
            If given, will insert the replace symbol with sites having symbols
            in this list
        """
        from random import choice
        if not swap_symbs:
            symbs = [atom.symbol for atom in self.atoms]
            swap_symbs = list(set(symbs))
        num_inserted = 0
        max_attempts = 10 * len(self.atoms)

        calc = self.atoms.get_calculator()
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

            system_changes = (indx, old_symb, symbol)
            if not self._no_constraint_violations([system_changes]):
                continue
            calc.update_cf(system_changes)
            num_inserted += 1
        self._build_atoms_list()
        calc.clear_history()

        if attempts == max_attempts:
            raise RuntimeError("Could insert {} {} atoms!".format(num, symbol))

    def update_current_energy(self):
        """Enforce a new energy evaluation."""
        self.current_energy = \
            self.atoms.get_calculator().calculate(None, None, None)
        self.bias_energy = 0.0
        for bias in self.bias_potentials:
            self.bias_energy += bias.calculate_from_scratch(self.atoms)
        self.current_energy += self.bias_energy

    def set_symbols(self, symbs):
        """Set the symbols of this Monte Carlo run.

        Parameters:

        symbs: list
            Symbols to insert. Has to have the same length as the
            attached atoms object.
        """
        self.atoms.get_calculator().set_symbols(symbs)
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
            raise TooFewElementsError(
                "There is only one element in the given atoms object!")
        n_elems_more_than_2 = 0
        for _, value in count.items():
            if value >= 2:
                n_elems_more_than_2 += 1
        if n_elems_more_than_2 < 2:
            raise TooFewElementsError(
                "There is only one element that has more than one atom")

    def log(self, msg, mode="info"):
        """Logs the message as info."""
        allowed_modes = ["info", "warning"]
        if mode not in allowed_modes:
            raise ValueError("Mode has to be one of {}".format(allowed_modes))

        _logger(msg)

    def _no_constraint_violations(self, system_changes):
        """Check if the proposed moves violate any of the constraints.

        Parameters:

        system_changes: list
            Changes of the proposed move
        """
        for constraint in self.constraints:
            if not constraint(system_changes):
                return False
        return True

    def reset(self):
        """Reset all member variables to their original values."""
        for _, obs in self.observers:
            obs.reset()

        self.filter.reset()
        self.current_step = 0
        self.num_accepted = 0
        self.mean_energy.clear()
        self.energy_squared.clear()
        self.corrtime_energies = []

    def _build_atoms_list(self):
        """
        Create a dictionary of the indices of each atom which is used to
        make sure that two equal atoms cannot be swapped
        """
        self.atoms_tracker.init_tracker(self.atoms)
        self.symbols = self.atoms_tracker.symbols

    def _update_tracker(self, system_changes):
        """Update the atom tracker."""
        self.atoms_tracker.update_swap_move(system_changes)

    def add_constraint(self, constraint):
        """Add a new constraint to the sampler.

        Parameters:

        constraint: MCConstraint
            Instance of a constraint object
        """
        self.constraints.append(constraint)

    def add_bias(self, potential):
        """Add a new bias potential.

        Parameters:

        potential: Bias potential
            Potential to be added
        """
        if not isinstance(potential, BiasPotential):
            raise TypeError("potential has to be of type BiasPotential")
        self.bias_potentials.append(potential)

    def attach(self, obs, interval=1):
        """Attach observers to be called on a given MC step interval.

        Parameters:

        obs: MCObserver
            Observer to be added

        interval: int
            How often the observer should be called
        """
        if (callable(obs)):
            self.observers.append((interval, obs))
        else:
            raise ValueError("The observer has to be a callable class!")

    def run(self, steps=100):
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

        # self.current_step gets updated in the _mc_step function
        self.reset()

        # Probe bias energy and remove bias
        self._probe_energy_bias()
        self.reset()

        start = time.time()
        prev = self.current_step
        while(self.current_step < steps):
            E, _ = self._mc_step()

            self.mean_energy += E
            self.energy_squared += E**2

            if (time.time() - start > self.status_every_sec):
                ms_per_step = 1000.0 * self.status_every_sec / \
                    float(self.current_step - prev)
                accept_rate = self.num_accepted / float(self.current_step)
                self.log(
                    "%d of %d steps. %.2f ms per step. Acceptance rate: %.2f" %
                    (self.current_step, steps, ms_per_step, accept_rate))
                prev = self.current_step
                start = time.time()

            if self.quit:
                break

        self.log(
            "Reached maximum number of steps ({} mc steps)".format(steps))

        # NOTE: Also update current_energy
        self._undo_energy_bias_from_eci()

    @property
    def meta_info(self):
        """Return dict with meta info."""
        import sys
        import datetime
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        v_info = sys.version_info
        meta_info = {
            "timestamp": st,
            "python_version": "{}.{}.{}".format(v_info.major, v_info.minor,
                                                v_info.micro)
        }
        return meta_info

    def get_thermodynamic_quantities(self):
        """Compute thermodynamic quantities."""
        quantities = {}
        mean_energy = self.mean_energy.mean
        quantities["energy"] = mean_energy + self.energy_bias
        mean_sq = self.energy_squared.mean
        quantities["heat_capacity"] = \
            (mean_sq - mean_energy**2) / (kB * self.T**2)
        quantities["energy_var"] = mean_sq - mean_energy**2
        quantities["temperature"] = self.T
        at_count = self.count_atoms()
        for key, value in at_count.items():
            name = "{}_conc".format(key)
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
        n = len(self.atoms)
        self.rand_a = np.random.randint(0, n)
        self.rand_b = np.random.randint(0, n)
        symb_a = self.symbols[np.random.randint(0, len(self.symbols))]
        symb_b = symb_a
        while (symb_b == symb_a):
            symb_b = self.symbols[np.random.randint(0, len(self.symbols))]

        rand_pos_a = self.atoms_tracker.get_random_indx_of_symbol(symb_a)
        rand_pos_b = self.atoms_tracker.get_random_indx_of_symbol(symb_b)
        system_changes = [(rand_pos_a, symb_a, symb_b),
                          (rand_pos_b, symb_b, symb_a)]
        return system_changes

    def _accept(self, system_changes):
        """Return `True` if the trial step is accepted.

        Parameters:

        system_changes: list
            List with system changes
        """
        self.last_energies[0] = self.current_energy

        # NOTE: Calculate updates the system
        calc = self.atoms.get_calculator()
        self.new_energy = calc.get_energy_given_change(system_changes)

        # NOTE: As this is called after calculate, the changes has
        # already been introduced to the system
        self.new_bias_energy = 0.0
        for bias in self.bias_potentials:
            self.new_bias_energy += bias(system_changes)
        self.new_energy += self.new_bias_energy
        self.last_energies[1] = self.new_energy

        # Standard Metropolis acceptance criteria
        if (self.new_energy < self.current_energy):
            return True
        kT = self.T * kB
        energy_diff = self.new_energy - self.current_energy
        probability = np.exp(-energy_diff / kT)
        return np.random.rand() <= probability

    def count_atoms(self):
        """Count the number of each element."""
        atom_count = {key: 0 for key in self.symbols}
        for atom in self.atoms:
            atom_count[atom.symbol] += 1
        return atom_count

    def _mc_step(self):
        """Make one Monte Carlo step by swithing two atoms."""
        self.current_step += 1
        self.last_energies[0] = self.current_energy

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
            raise CanNotFindLegalMoveError(msg)

        move_accepted = self._accept(system_changes)

        # At this point the new energy is calculated in the _accept function
        self.last_energies[1] = self.new_energy

        if (move_accepted):
            self.current_energy = self.new_energy
            self.bias_energy = self.new_bias_energy
            self.num_accepted += 1
        else:
            # Reset the sytem back to original
            for change in system_changes:
                indx = change[0]
                old_symb = change[1]
                assert (self.atoms[indx].symbol == change[2])
                self.atoms[indx].symbol = old_symb

        if (move_accepted):
            self.atoms.get_calculator().clear_history()
        else:
            self.atoms.get_calculator().restore()

        if (move_accepted):
            # Update the atom_indices
            self._update_tracker(system_changes)
        else:
            new_symb_changes = []
            for change in system_changes:
                new_symb_changes.append((change[0], change[1], change[1]))
            system_changes = new_symb_changes

        # Execute all observers
        for entry in self.observers:
            interval = entry[0]
            if (self.current_step % interval == 0):
                obs = entry[1]
                obs(system_changes)
        self.filter.add(self.current_energy)
        return self.current_energy, move_accepted
