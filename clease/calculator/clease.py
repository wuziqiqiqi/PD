"""Calculator for Cluster Expansion."""
import sys
import numpy as np
from ase.utils import basestring
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from clease import CorrFunction
from clease.settings import ClusterExpansionSetting
# from clease.calculator.duplication_count_tracker import DuplicationCountTracker
from clease_cxx import PyCEUpdater


class MovedIgnoredAtomError(Exception):
    """Raised when ignored atoms is moved."""
    pass


class Clease(Calculator):
    """Class for calculating energy using CLEASE.

    Parameters:

    setting: `ClusterExpansionSetting` object

    eci: dict
        Dictionary containing cluster names and their ECI values

    init_cf: `None` or dictionary (optional)
        If the correlation function of Atoms object is known, one can supply
        its correlation function values such that the initial assessment step
        is skipped. The dictionary should contain cluster names (same as the
        ones provided in `eci`) and their correlation function values.

    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.
    """

    name = 'CLEASE'
    implemented_properties = ['energy']

    def __init__(self, setting, eci=None, init_cf=None,
                 logfile=None):
        Calculator.__init__(self)

        if not isinstance(setting, ClusterExpansionSetting):
            msg = "setting must be CEBulk or CECrystal object."
            raise TypeError(msg)
        self.parameters["eci"] = eci
        self.setting = setting
        self.corrFunc = CorrFunction(setting)
        self.eci = eci
        # store cluster names
        self.cf_names = list(eci.keys())

        # calculate init_cf or convert init_cf to array
        if init_cf is None or isinstance(init_cf, dict):
            self.init_cf = init_cf
        else:
            msg = "'init_cf' must be a dictionary containing cluster names "
            msg += "and their correlation function values."
            raise TypeError(msg)

        if self.init_cf is not None and len(self.eci) != len(self.init_cf):
            msg = "length of provided ECIs and correlation functions do not "
            msg += "match"
            raise ValueError(msg)

        # logfile
        if isinstance(logfile, basestring):
            if logfile == '-':
                logfile = sys.stdout
            else:
                logfile = open(logfile, 'a')
        self.logfile = logfile

        self.energy = None
        # reference atoms for calculating the cf and energy for new atoms
        self.atoms = None
        self.symmetry_group = None
        self.is_backround_index = None

        # C++ updater initialized when atoms are set
        self.updater = None

    def _set_norm_factors(self):
        """Set normalization factor for each cluster.

        The normalization factor only kicks in when the cell is too small and
        thus, include self-interactions. This methods corrects the impact of
        self-interactions.
        """
        symm_group = np.zeros(len(self.setting.atoms), dtype=np.uint8)
        for num, group in enumerate(self.setting.index_by_trans_symm):
            symm_group[group] = num

        cluster_list = self.setting.cluster_list
        for cluster in cluster_list:
            fig_keys = list(set(cluster.get_all_figure_keys()))
            num_occ = {}
            for key in fig_keys:
                num_occ[key] = cluster_list.num_occ_figure(key,
                    cluster.name, symm_group, self.setting.trans_matrix)
            num_fig_occ = cluster.num_fig_occurences
            norm_factors = {}
            for key in fig_keys:
                tot_num = num_occ[key]
                num_unique = len(set(key.split("-")))
                norm_factors[key] = \
                    float(tot_num)/(num_unique*num_fig_occ[key])

            norm_factor_list = []
            for fig in cluster.indices:
                key = cluster.get_figure_key(fig)
                norm_factor_list.append(norm_factors[key])
            cluster.info['normalization_factor'] = norm_factor_list

    def set_atoms(self, atoms):
        self.atoms = atoms

        if self.init_cf is None:
            self.init_cf = \
                self.corrFunc.get_cf_by_names(self.atoms, self.cf_names)

        if len(self.setting.atoms) != len(atoms):
            msg = "Passed Atoms object and setting.atoms should have same "
            msg += "number of atoms."
            raise ValueError(msg)

        if not np.allclose(atoms.positions, self.setting.atoms.positions):
            msg = "Positions of all atoms in the passed Atoms object and "
            msg += "setting.atoms should be the same. "
            raise ValueError(msg)

        self.symmetry_group = np.zeros(len(atoms), dtype=int)
        for symm, indices in enumerate(self.setting.index_by_trans_symm):
            self.symmetry_group[indices] = symm
        self.is_backround_index = np.zeros(len(atoms), dtype=np.uint8)
        self.is_backround_index[self.setting.background_indices] = 1

        self._set_norm_factors()
        self.updater = PyCEUpdater(self.atoms, self.setting, self.init_cf,
                                   self.eci, self.setting.cluster_list)

    def get_energy_given_change(self, system_changes):
        """
        Calculate the energy when the change is known. No checking will be
        performed.

        Parameters:

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        self.update_cf(system_changes=system_changes)
        self.energy = self.updater.get_energy()
        self.results['energy'] = self.energy
        return self.energy

    def calculate(self, atoms, properties, system_changes):
        """Calculate the energy of the passed Atoms object.

        If accept=True, the most recently used atoms object is used as a
        reference structure to calculate the energy of the passed atoms.
        Returns energy.

        Parameters:

        atoms: Atoms object
            ASE Atoms object

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        Calculator.calculate(self, atoms)
        self.update_energy()
        self.energy = self.updater.get_energy()
        self.results['energy'] = self.energy
        return self.energy

    def clear_history(self):
        self.updater.clear_history()

    def restore(self):
        """Restore the Atoms object to its original configuration and energy.

        This method reverts the Atoms object to its oldest state stored in
        memory. The state is restored to either
        (1) an initial state when the calculator was attached, or
        (2) the state at which the `clear_history()` method was invoked last
            time.

        NOTE: The maximum capacity for the history buffer is 1000 steps
        """
        self.updater.undo_changes()
        self.energy = self.updater.get_energy()
        self.results['energy'] = self.energy

    def update_energy(self):
        """Update correlation function and get new energy."""
        self.update_cf()
        self.energy = self.updater.get_energy()

    @property
    def indices_of_changed_atoms(self):
        """Return the indices of atoms that have been changed."""
        changed = self.updater.get_changed_sites(self.atoms)
        for index in changed:
            if self.is_backround_index[index] and \
                    self.setting.ignore_background_atoms:
                msg = "Atom with index {} is a background atom.".format(index)
                raise MovedIgnoredAtomError(msg)

        return changed

    def get_cf(self):
        """Return the correlation functions as a dict"""
        return self.updater.get_cf()

    def update_cf(self, system_changes=None):
        """Update correlation function based on the reference value.

        Paramters:

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        if system_changes is None:
            swapped_indices = self.indices_of_changed_atoms
            symbols = self.updater.get_symbols()
            system_changes = [(x, symbols[x], self.atoms[x].symbol)
                              for x in swapped_indices]
        for change in system_changes:
            self.updater.update_cf(change)

    @property
    def cf(self):
        temp_cf = self.updater.get_cf()
        return [temp_cf[x] for x in self.cf_names]

    def log(self):
        """Write energy to log file."""
        if self.logfile is None:
            return True
        self.logfile.write('{}\n'.format(self.energy))
        self.logfile.flush()

    def update_eci(self, eci):
        """Update the ECI values.

        Parameters:

        eci: dict
            Dictionary with new ECIs
        """
        self.eci = eci
        self.updater.set_eci(eci)

    def get_singlets(self):
        return self.updater.get_singlets()
