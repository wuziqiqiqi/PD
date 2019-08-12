# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
from __future__ import division
import sys
import numpy as np
import ase.units as units
import time
import logging
from scipy import stats
from ase.units import kJ, mol
from clease.montecarlo.exponential_filter import ExponentialFilter
from clease.montecarlo.averager import Averager
from clease.montecarlo.util import get_new_state
from clease.montecarlo import BiasPotential
from clease.montecarlo .swap_move_index_tracker import SwapMoveIndexTracker


class DidNotReachEquillibriumError(Exception):
    pass


class TooFewElementsError(Exception):
    pass


class CanNotFindLegalMoveError(Exception):
    pass


class Montecarlo(object):
    """
    Class for running Monte Carlo at fixed composition

    :param Atoms atoms: ASE atoms object (with CE calculator attached!)
    :param float temp: Temperature of Monte Carlo simulation in Kelvin
    :param list indeces: Atoms involved Monte Carlo swaps. default is all
                    atoms (currently this has no effect!).
    :param str logfile: Filename for logging (default is logging to console)
    :param bool plot_debug: If True it will create some diagnositc plots during
                       equilibration
    :param bool recycle_waste: If True also rejected states will be used to
        estimate averages
    :param int max_constraint_attempts: Maximum number of allowed attempts to
        for finding a candidate move that does not violate the constraints.
        A CanNotFindLegalMoveError is raised if the maximum number of attempts
        is reaced.
    :param bool accept_first_trial_move_after_reset: If True the first trial
        move after reset and set_symbols will be accepted
    """

    def __init__(self, atoms, temp):
        self.name = "MonteCarlo"
        self.atoms = atoms
        self.T = temp

        # List of observers that will be called every n-th step
        # similar to the ones used in the optimization routines
        self.observers = []

        self.constraints = []
        self.max_allowed_constraint_pass_attempts = max_constraint_attempts

        if self.max_allowed_constraint_pass_attempts <= 0:
            raise ValueError("Max. constraint attempts has to be > 0!")

        self.bias_potentials = []

        self.current_step = 0
        self.num_accepted = 0
        self.status_every_sec = 30
        self.atoms_tracker = SwapMoveIndexTracker()
        self.symbols = []
        self._build_atoms_list()
        E0 = self.atoms.get_calculator().get_potential_energy()
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

        # Array of energies used to estimate the correlation time
        self.corrtime_energies = []
        self.correlation_info = None
        self.plot_debug = plot_debug

        self.filter = ExponentialFilter(min_time=0.2*len(self.atoms),
                                        max_time=20*len(self.atoms),
                                        n_subfilters=10)

    def _probe_energy_bias(self, num_steps=1000):
        """
        Run MC steps to probe the energy bias. The bias
        will be subtracted off the zeroth ECI and then
        added to the total energy durin gpost processing.
        """
        self.log("Probing energy bias using {} MC steps...".format(num_steps))
        for _ in range(num_steps):
            self._mc_step()

        self.log("Energy after probing: {}".format(self.current_energy))
        self.energy_bias = self.current_energy
        self._remove_bias_from_empty_eci(self.energy_bias)

    def _remove_bias_from_empty_eci(self, bias):
        """
        Remove the energy bias from the zeroth ECI.

        :param float: Energy bias
        """
        eci = self.atoms.get_calculator().eci
        c0_eci = eci['c0']
        c0_eci -= bias/len(self.atoms)
        eci['c0'] = c0_eci

        self.atoms.get_calculator().update_ecis(eci)
        self.current_energy = self.atoms.get_calculator().get_energy()
        self.last_energies[0] = self.current_energy

        if abs(self.current_energy) > 1E-6:
            raise RuntimeError("Energy is not 0 after subtracting "
                               "the bias. Got {}".format(self.current_energy))

        self.log('Bias subtracted from empty cluster...')

    def _undo_energy_bias_from_eci(self):
        eci = self.atoms.get_calculator().eci
        eci['c0'] += self.energy_bias/len(self.atoms)
        self.atoms.get_calculator().update_ecis(eci)
        self.log('Empty cluster ECI reset to original value...')

    def insert_symbol(self, symb, indices):
        """Insert symbols on a predefined set of indices.

        :param str symb: Symbol to be inserted
        :param list indices: Indices where symb should be inserted
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

        :param str symbol: Symbol to insert
        :param int num: Number of sites
        :param list swap_symbs: If given, will insert replace symbol with sites
                          having symbols in this list
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
        self.current_energy = self.atoms.get_calculator().get_energy()
        self.bias_energy = 0.0
        for bias in self.bias_potentials:
            self.bias_energy += bias.calculate_from_scratch(self.atoms)
        self.current_energy += self.bias_energy

    def set_symbols(self, symbs):
        """Set the symbols of this Monte Carlo run.

        :param list: Symbols to insert. Has to have the same length as the
            attached atoms object
        """
        self.atoms.get_calculator().set_symbols(symbs)
        self._build_atoms_list()
        self.update_current_energy()

        if self.accept_first_trial_move_after_reset:
            self.is_first = True

    def _check_symbols(self):
        """
        Checks that there is at least to different symbols
        """
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
        for key, value in count.items():
            if value >= 2:
                n_elems_more_than_2 += 1
        if n_elems_more_than_2 < 2:
            raise TooFewElementsError(
                "There is only one element that has more than one atom")

    def log(self, msg, mode="info"):
        """
        Logs the message as info
        """
        allowed_modes = ["info", "warning"]
        if mode not in allowed_modes:
            raise ValueError("Mode has to be one of {}".format(allowed_modes))

        if mode == "info":
            self.logger.info(msg)
        elif mode == "warning":
            self.logger.warning(msg)

    def _no_constraint_violations(self, system_changes):
        """
        Checks if the proposed moves violates any of the constraints

        :param list system_changes: Changes of the proposed move
            see :py:class:`cemc.mcmc.mc_observers.MCObserver`

        :return: True/False if True then no constraints are violated
        :rtype: bool
        """
        for constraint in self.constraints:
            if not constraint(system_changes):
                return False
        return True

    def reset(self):
        """
        Reset all member variables to their original values
        """
        for interval, obs in self.observers:
            obs.reset()

        self.filter.reset()
        self.current_step = 0
        self.num_accepted = 0
        self.mean_energy.clear()
        self.energy_squared.clear()
        self.corrtime_energies = []

    def _build_atoms_list(self):
        """
        Creates a dictionary of the indices of each atom which is used to
        make sure that two equal atoms cannot be swapped
        """
        self.atoms_tracker.init_tracker(self.atoms)
        self.symbols = self.atoms_tracker.symbols

    def _update_tracker(self, system_changes):
        """
        Update the atom tracker
        """
        self.atoms_tracker.update_swap_move(system_changes)

    def add_constraint(self, constraint):
        """
        Add a new constraint to the sampler

        :param MCConstraint constraint: Constraint
        """
        self.constraints.append(constraint)

    def add_bias(self, potential):
        """Add a new bias potential.

        :param BiasPotential potential: Potential to be added
        """
        if not isinstance(potential, BiasPotential):
            raise TypeError("potential has to be of type BiasPotential")
        self.bias_potentials.append(potential)

    def attach(self, obs, interval=1):
        """
        Attach observers that is called on each MC step
        and receives information of which atoms get swapped

        :param MCObserver obs: Observer to be added
        :param int interval: the obs.__call__ method is called at mc steps
                         separated by interval
        """
        if (callable(obs)):
            self.observers.append((interval, obs))
        else:
            raise ValueError("The observer has to be a callable class!")

    def run(self, steps=100):
        """Run Monte Carlo simulation

        :param int steps: Number of steps in the MC simulation
        """
        # Check the number of different elements are correct to avoid
        # infinite loops
        self._check_symbols()

        self.update_current_energy()

        # Atoms object should have attached calculator
        # Add check that this is show
        self._mc_step()

        # self.current_step gets updated in the _mc_step function
        log_status_conv = True
        self.reset()

        # Probe bias energy and remove bias
        self._probe_energy_bias()
        self.reset()

        while(self.current_step < steps):
            E, accept = self._mc_step(verbose=verbose)

            self.mean_energy += E
            self.energy_squared += E_sq

            if (time.time() - start > self.status_every_sec):
                ms_per_step = 1000.0 * self.status_every_sec / \
                    float(self.current_step - prev)
                accept_rate = self.num_accepted / float(self.current_step)
                log_status_conv = True
                self.log(
                    "%d of %d steps. %.2f ms per step. Acceptance rate: %.2f" %
                    (self.current_step, steps, ms_per_step, accept_rate))
                prev = self.current_step
                start = time.time()

        self.log(
            "Reached maximum number of steps ({} mc steps)".format(steps))

        # NOTE: Does not reset the energy bias to 0
        self._undo_energy_bias_from_eci()
        return totalenergies

    @property
    def meta_info(self):
        """Return dict with meta info."""
        import sys
        import datetime
        import time
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        v_info = sys.version_info
        meta_info = {
            "timestamp": st,
            "python_version": "{}.{}.{}".format(v_info.major, v_info.minor,
                                                v_info.micro)
        }
        return meta_info

    def get_thermodynamic(self):
        """
        Compute thermodynamic quantities

        :return: thermodynamic data
        :rtype: dict
        """
        quantities = {}
        mean_energy = self.mean_energy.mean
        quantities["energy"] = mean_energy + self.energy_bias
        mean_sq = self.energy_squared.mean
        quantities["heat_capacity"] = (
            mean_sq - mean_energy**2) / (units.kB * self.T**2)
        quantities["energy_std"] = np.sqrt(self._get_var_average_energy())
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

        :return: Trial move
        :rtype: list
        """
        self.rand_a = self.indeces[np.random.randint(0, len(self.indeces))]
        self.rand_b = self.indeces[np.random.randint(0, len(self.indeces))]
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
        """
        Returns True if the trial step is accepted

        :param list system_changes: List with the proposed system changes

        :return: True/False if the move is accepted or not
        :rtype: bool
        """
        self.last_energies[0] = self.current_energy

        # NOTE: Calculate updates the system
        self.new_energy = self.atoms.get_calculator().calculate(
            self.atoms, ["energy"], system_changes)

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
        kT = self.T * units.kB
        energy_diff = self.new_energy - self.current_energy
        probability = np.exp(-energy_diff / kT)
        return np.random.rand() <= probability

    def count_atoms(self):
        """
        Count the number of each species

        :return: Number of each species
        :rtype: dict
        """
        atom_count = {key: 0 for key in self.symbols}
        for atom in self.atoms:
            atom_count[atom.symbol] += 1
        return atom_count

    def _mc_step(self, verbose=False):
        """
        Make one Monte Carlo step by swithing two atoms
        """
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
            self.atoms.get_calculator().undo_changes()

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
