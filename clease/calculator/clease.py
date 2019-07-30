"""Calculator for Cluster Expansion."""
import sys
from copy import deepcopy
import numpy as np
from ase.utils import basestring
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from clease import CorrFunction, CEBulk, CECrystal
from clease.corrFunc import equivalent_deco
from clease.tools import get_sparse_column_matrix, symbols2integer
from clease.tools import bf2npyarray
from clease.jit import jit
from clease.calculator.duplication_count_tracker import DuplicationCountTracker
from clease_cxx import PyCEUpdater


class MovedIgnoredAtomError(Exception):
    """Raised when ignored atoms is moved."""
    pass


class Clease(Calculator):
    """Class for calculating energy using CLEASE.

    Arguments:
    =========
    setting: CEBulk or BulkSapcegroup object

    cluster_name_eci: dictionary of list of tuples containing
                      cluster names and ECI

    init_cf: (optional) correlation function of init_cf

    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.
    """

    name = 'CLEASE'
    implemented_properties = ['energy']

    def __init__(self, setting, cluster_name_eci=None, init_cf=None,
                 logfile=None):
        Calculator.__init__(self)

        if not isinstance(setting, (CEBulk, CECrystal)):
            msg = "setting must be CEBulk or CECrystal object."
            raise TypeError(msg)
        self.parameters["eci"] = cluster_name_eci
        self.setting = setting
        self.CF = CorrFunction(setting)

        # check cluster_name_eci and separate them out
        if isinstance(cluster_name_eci, list) and \
           (all(isinstance(i, tuple) for i in cluster_name_eci) or
                all(isinstance(i, list) for i in cluster_name_eci)):
            self.cluster_names = [tup[0] for tup in cluster_name_eci]
            self.eci = np.array([tup[1] for tup in cluster_name_eci])
        elif isinstance(cluster_name_eci, dict):
            self.cluster_names = []
            self.eci = []
            for cluster_name, eci in cluster_name_eci.items():
                self.cluster_names.append(cluster_name)
                self.eci.append(eci)
            self.eci = np.array(self.eci)
        else:
            msg = "'cluster_name_eci' needs to be either (1) a list of tuples "
            msg += "or (2) a dictionary.\n They can be etrieved by "
            msg += "'get_cluster_name_eci' method in Evaluate class."
            raise TypeError(msg)

        # calculate init_cf or convert init_cf to array
        if init_cf is None:
            self.init_cf = init_cf
        elif isinstance(init_cf, list):
            if all(isinstance(i, (tuple, list)) for i in init_cf):
                cluster_names = [tup[0] for tup in init_cf]
                # cluster_name_eci and init_cf in the same order
                if cluster_names == self.cluster_names:
                    self.init_cf = np.array([tup[1] for tup in init_cf],
                                             dtype=float)
                # not in the same order
                else:
                    self.init_cf = []
                    for name in self.cluster_names:
                        indx = cluster_names.index(name)
                        self.init_cf.append(init_cf[indx][1])
                    self.init_cf = np.array(self.init_cf, dtype=float)
            else:
                self.init_cf = np.array(init_cf, dtype=float)
        elif isinstance(init_cf, dict):
            self.init_cf = np.array([init_cf[x] for x in self.cluster_names],
                               dtype=float)
        else:
            raise TypeError("'init_cf' needs to be either (1) a list "
                            "of tuples, (2) a dictionary, or (3) numpy array "
                            "containing correlation function in the same "
                            "order as the 'cluster_name_eci'.")

        if self.init_cf is not None and len(self.eci) != len(self.init_cf):
            raise ValueError('length of provided ECIs and correlation '
                             'functions do not match')

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
        self.dupl_tracker = DuplicationCountTracker(self.setting)

        self.equiv_deco = self._precalculate_equivalent_decorations()

        self.clusters_per_symm_group, self.one_body = \
            self._place_clusters_in_symm_groups()

        # C++ updater initialised when atoms are set
        self.updater = None

    def _get_cluster_info_with_dup_factors(self, cluster_info):
        info = []
        for all_info in cluster_info:
            info.append(all_info)

            for k in all_info.keys():
                cluster = info[-1][k]

                dup_factors = [self.dupl_tracker.factor(
                    cluster, non_trans, order)
                    for non_trans, order in zip(cluster["indices"],
                                                cluster["order"])]

                info[-1][k]["dup_factors"] = dup_factors
        return info

    def _precalculate_equivalent_decorations(self):
        equiv_decos = []

        for symm in range(0, len(self.setting.cluster_info)):
            equiv_decos.append({})
            for name in self.cluster_names:
                prefix = (name.rpartition('_')[0])
                if prefix not in self.setting.cluster_info[symm].keys():
                    continue

                dec_str = name.rpartition('_')[-1]
                dec = [int(x) for x in dec_str]
                cluster = self.setting.cluster_info[symm][prefix]
                equiv_deco = np.array(
                            equivalent_deco(dec, cluster["equiv_sites"]),
                            dtype=np.int32)
                equiv_decos[-1][name] = equiv_deco
        return equiv_decos

    def _place_clusters_in_symm_groups(self):
        clst_per_symm_group = []
        for info in self.setting.cluster_info:
            clst_in_symm_group = []
            for cf_num, name in enumerate(self.cluster_names):
                prefix = name.rpartition('_')[0]

                if prefix in info.keys() and prefix[:2] not in ['c0', 'c1']:
                    clst_in_symm_group.append((cf_num, name))
            clst_per_symm_group.append(clst_in_symm_group)

        one_body = []
        for i, name in enumerate(self.cluster_names):
            if name.startswith('c1'):
                one_body.append((i, name))
        return clst_per_symm_group, one_body

    def set_atoms(self, atoms):
        self.atoms = atoms
        # self.setting.set_active_template(atoms=atoms)
        # self.dupl_tracker = DuplicationCountTracker(self.setting)

        if self.init_cf is None:
            self.init_cf = self.CF.get_cf_by_cluster_names(
                self.atoms, self.cluster_names, return_type='array')

        if len(self.setting.atoms) != len(atoms):
            msg = "Passed Atoms object and setting.atoms should have "
            msg += "same number of atoms."
            raise ValueError(msg)
        if not np.allclose(atoms.positions, self.setting.atoms.positions):
            msg = "atomic positions of all atoms in the passed Atoms "
            msg += "object and setting.atoms should be the same. "
            raise ValueError(msg)
        self.symmetry_group = np.zeros(len(atoms), dtype=int)
        for symm, indices in enumerate(self.setting.index_by_trans_symm):
            self.symmetry_group[indices] = symm
        self.is_backround_index = np.zeros(len(atoms), dtype=np.uint8)
        self.is_backround_index[self.setting.background_indices] = 1

        cf_dict = dict(zip(self.cluster_names, self.init_cf))

        info = self._get_cluster_info_with_dup_factors(
            self.setting.cluster_info)
        self.updater = PyCEUpdater(
            self.atoms, self.setting, cf_dict,
            dict(zip(self.cluster_names, self.eci)), info)

    def get_energy_given_change(self, atoms, system_changes):
        """
        Calculate the energy when the change is known. No
        checking will be performed
        """
        self.update_cf(system_changes)
        self.energy = self.updater.get_energy()
        self.results['energy'] = self.energy
        return self.energy

    def calculate(self, atoms, properties, system_changes):
        """Calculate the energy of the passed atoms object.

        If accept=True, the most recently used atoms object is used as a
        reference structure to calculate the energy of the passed atoms.
        Returns energy.
        """
        Calculator.calculate(self, atoms)
        self.update_energy()
        self.energy = self.updater.get_energy()
        self.results['energy'] = self.energy
        return self.energy

    def clear_history(self):
        self.updater.clear_history()

    def restore(self):
        """Restore the old atoms and correlation functions to the reference."""
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
                raise MovedIgnoredAtomError("Atom with index {} is a "
                                            "background atom."
                                            "".format(index))

        return changed

    def get_cf_dict(self):
        """Return the correlation functions as a dict"""
        return self.updater.get_cf()

    def get_cf_list_tup(self):
        """Return the correlation function as a list of tuples"""
        cf = self.updater.get_cf()
        return [(k, v) for k, v in cf.items()]

    def _generate_normalization_factor(self):
        """Return a dictionary with all the normalization factors."""
        norm_fact = {}
        for symm, item in enumerate(self.setting.cluster_info):
            num_atoms = len(self.setting.index_by_trans_symm[symm])
            for name, info in item.items():
                if name not in norm_fact.keys():
                    norm_fact[name] = len(info["indices"]) * num_atoms
                else:
                    norm_fact[name] += len(info["indices"]) * num_atoms
        return norm_fact

    def update_cf(self, system_changes=None):
        """Update correlation function based on the reference value."""
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
        return [temp_cf[x] for x in self.cluster_names]

    def _check_atoms(self, atoms):
        """Check to see if the passed atoms argument valid.

        This method checks that:
            - atoms argument is Atoms object,
            - atoms has the same size and atomic positions as
                (1) setting.atoms,
                (2) reference Atoms object.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('Passed argument is not Atoms object')
        if len(self.atoms) != len(atoms):
            raise ValueError('Passed atoms does not have the same size '
                             'as previous atoms')
        if not np.allclose(self.atoms.positions, atoms.positions):
            raise ValueError('Atomic postions of the passed atoms are '
                             'different from init_atoms')

    def log(self):
        """Write energy to log file."""
        if self.logfile is None:
            return True
        self.logfile.write('{}\n'.format(self.energy))
        self.logfile.flush()
