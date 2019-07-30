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
        self.norm_factor = self._generate_normalization_factor()

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
            self.cf = init_cf
        elif isinstance(init_cf, list):
            if all(isinstance(i, (tuple, list)) for i in init_cf):
                cluster_names = [tup[0] for tup in init_cf]
                # cluster_name_eci and init_cf in the same order
                if cluster_names == self.cluster_names:
                    self.cf = np.array([tup[1] for tup in init_cf],
                                       dtype=float)
                # not in the same order
                else:
                    self.cf = []
                    for name in self.cluster_names:
                        indx = cluster_names.index(name)
                        self.cf.append(init_cf[indx][1])
                    self.cf = np.array(self.cf, dtype=float)
            else:
                self.cf = np.array(init_cf, dtype=float)
        elif isinstance(init_cf, dict):
            self.cf = np.array([init_cf[x] for x in self.cluster_names],
                               dtype=float)
        else:
            raise TypeError("'init_cf' needs to be either (1) a list "
                            "of tuples, (2) a dictionary, or (3) numpy array "
                            "containing correlation function in the same "
                            "order as the 'cluster_name_eci'.")

        if self.cf is not None and len(self.eci) != len(self.cf):
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
        self.num_si = self._get_num_self_interactions()
        self.dupl_tracker = DuplicationCountTracker(self.setting)
        self.sp_trans_mat = get_sparse_column_matrix(self.setting.trans_matrix)
        self.symb_id = symbols2integer(self.setting.basis_functions)
        self.bf_npy = bf2npyarray(self.setting.basis_functions, self.symb_id)
        self.cluster_info_npy = \
            self._get_cluster_info_npy(deepcopy(self.setting.cluster_info))

        self.equiv_deco = self._precalculate_equivalent_decorations()

        self.clusters_per_symm_group, self.one_body = \
            self._place_clusters_in_symm_groups()

        # C++ updater initialised when atoms are set
        self.updater = None
     
    def _get_cluster_info_npy(self, cluster_info):
        info = []
        for all_info in cluster_info:
            info.append(all_info)

            # Now convert the critial structures to numpy arrays
            for k in all_info.keys():
                cluster = info[-1][k]

                dup_factors = [self.dupl_tracker.factor(
                    cluster, non_trans, order)
                    for non_trans, order in zip(cluster["indices"],
                                                cluster["order"])]

                info[-1][k]["dup_factors"] = np.array(dup_factors)

                sorted_indices = []
                for subcluster, order in zip(cluster["indices"], cluster["order"]):
                    all_indx = [cluster["ref_indx"]] + subcluster
                    all_indx = [all_indx[i] for i in order]
                    sorted_indices.append(all_indx)
                info[-1][k]["indices"] = np.array(sorted_indices, dtype=np.int32)
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
        self.atoms = atoms  # .copy()
        if self.cf is None:
            self.cf = self.CF.get_cf_by_cluster_names(self.atoms,
                                                      self.cluster_names,
                                                      return_type='array')

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

        self.updater = PyCEUpdater(self.atoms, self.setting,
                                   self.get_cf_dict(),
                                   dict(zip(self.cluster_names, self.eci)))

    def calculate(self, atoms, properties, system_changes):
        """Calculate the energy of the passed atoms object.

        If accept=True, the most recently used atoms object is used as a
        reference structure to calculate the energy of the passed atoms.
        Returns energy.
        """
        self._check_atoms(atoms)
        Calculator.calculate(self, atoms)
        swapped_indices = self.update_energy()
        self.results['energy'] = self.energy
        self.log()
        if len(swapped_indices) == 0:
            return self.energy

        return self.energy

    def clear_history(self):
        self.updater.clear_history()

    def restore(self):
        """Restore the old atoms and correlation functions to the reference."""
        self.updater.undo_changes()

    def update_energy(self):
        """Update correlation function and get new energy."""
        swapped_indices = self.update_cf()
        self.energy = self.eci.dot(self.cf) * len(self.atoms)
        return swapped_indices

    @property
    def indices_of_changed_atoms(self):
        """Return the indices of atoms that have been changed."""
        #old_numbers = 
        # o_numbers = self.ref_atoms.numbers
        # n_numbers = self.atoms.numbers
        # check = (n_numbers == o_numbers)
        # changed = np.argwhere(check == 0)[:, 0]
        # changed = np.unique(changed)
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
        return dict(self.get_cf_list_tup())

    def get_cf_list_tup(self):
        """Return the correlation function as a list of tuples"""
        return zip(self.cluster_names, self.cf)

    def _symbol_by_index(self, indx):
        return [self.ref_atoms[indx].symbol, self.atoms[indx].symbol]

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

    def update_cf(self):
        """Update correlation function based on the reference value."""
        swapped_indices = self.indices_of_changed_atoms
        self.cf = deepcopy(self.ref_cf)
        new_symbs = {}
        # Reset the atoms object
        for indx in swapped_indices:
            new_symbs[indx] = self.atoms[indx].symbol
            self.atoms[indx].symbol = self.ref_atoms[indx].symbol

        atoms_npy = np.array([self.symb_id.get(atom.symbol, -1)
                              for atom in self.atoms], dtype=np.int32)
        for indx in swapped_indices:
            # Swap one index at the time
            self.atoms[indx].symbol = new_symbs[indx]
            new_symbid = self.symb_id[new_symbs[indx]]
            symm = self.symmetry_group[indx]

            # Update one_body
            for item in self.one_body:
                i = item[0]
                name = item[1]
                dec = int(name[-1])
                self.cf[i] += (self.bf_npy[dec, new_symbid] -
                               self.bf_npy[dec, atoms_npy[indx]])\
                    / len(atoms_npy)

            for item in self.clusters_per_symm_group[symm]:
                i = item[0]
                name = item[1]
                # find c{num} in cluster type
                n = int(name[1])

                prefix = name.rpartition('_')[0]
                dec_str = name.rpartition('_')[-1]
                dec = [int(x) for x in dec_str]

                cluster_npy = self.cluster_info_npy[symm][prefix]
                count = self.norm_factor[prefix]
                cf_tot = self.cf[i] * count

                # Try to use JIT function
                dup_factors = cluster_npy["dup_factors"]
                equiv_deco = self.equiv_deco[symm][name]
                indices = cluster_npy["indices"]

                cf_change = cf_change_by_indx_jit(
                    atoms_npy, indx, new_symbid, indices, equiv_deco,
                    dup_factors, self.sp_trans_mat, self.bf_npy)

                cf_change /= self.num_si[prefix]
                self.cf[i] = (cf_tot + (n * cf_change)) / count

            # Update the number array, JIT version assumes that this array is
            # updated after the CF change is requested
            atoms_npy[indx] = new_symbid
        return swapped_indices

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
        if len(self.ref_atoms) != len(atoms):
            raise ValueError('Passed atoms does not have the same size '
                             'as previous atoms')
        if not np.allclose(self.ref_atoms.positions, atoms.positions):
            raise ValueError('Atomic postions of the passed atoms are '
                             'different from init_atoms')

    def log(self):
        """Write energy to log file."""
        if self.logfile is None:
            return True
        self.logfile.write('{}\n'.format(self.energy))
        self.logfile.flush()

    def _get_num_self_interactions(self):
        num_si = {}
        for item in self.setting.cluster_info:
            for name, info in item.items():
                if info["size"] <= 1:
                    num_si[name] = 1.
                    continue
                ref_indx = info["ref_indx"]
                num_int = sum(sub.count(ref_indx) for sub in info["indices"])
                num_equiv_indices = 0
                for sub in info["indices"]:
                    bin_count = np.bincount(sub)
                    bin_count[bin_count > 0] -= 1
                    num_equiv_indices += np.sum(bin_count)
                num_int += num_equiv_indices + len(info["indices"])
                num_si[name] = float(num_int)/len(info["indices"])
                num_si[name] = 1.0
        return num_si


def list2str(array):
    return "-".join(str(x) for x in array)


class DuplicationCountTracker(object):
    """Tracks duplication counts and normalization factors.

    Arguments
    ==========
    cluster_info: list of dicts
        The entire info entry in settings
    """
    def __init__(self, setting):
        self.symm_group = np.zeros(len(setting.atoms), dtype=np.uint8)
        for num, group in enumerate(setting.index_by_trans_symm):
            self.symm_group[group] = num

        self.cluster_info = setting.cluster_info
        self.trans_matrix = setting.trans_matrix
        self.occ_count_all = self._occurence_count_all_symm_groups()
        self._norm_factors = self._get_norm_factors()

    def factor(self, cluster, indices, order):
        """Get the normalization factor to correct for self interactions.

        Arguments
        ===========
        cluster: dict
            Dictionary holding information about the cluster

        indices: list
            Indices of the particular sub cluster

        order: list
            Order of the indices in the sub cluster
        """
        key = self.index_key(cluster["ref_indx"], indices, order,
                             cluster["equiv_sites"])
        return self._norm_factors[cluster["symm_group"]][cluster["name"]][key]

    def index_key(self, ref_index, indices, order, equiv_sites):
        """Return a string representing the key for a given order.

        Arguments
        ==========
        ref_index: int
            Reference index

        indices: list
            List representing the indices in a sub-cluster

        order: list
            Order of the indices in the sub cluster
        """
        index_with_ref = [ref_index] + indices
        srt_indices = [index_with_ref[i] for i in order]
        return list2str(self._order_equiv_sites(equiv_sites, srt_indices))

    def _get_norm_factors(self):
        """Calculate all normalization factors."""
        norm_factors = []
        for item in self.cluster_info:
            factors = self._get_norm_factors_per_symm_group(item)
            norm_factors.append(factors)
        return norm_factors

    def _get_norm_factors_per_symm_group(self, clusters):
        """Get normalization factors per symmetry group.

        Arguments:
        =========
        clusters: dict
            Information dict about all clusters in a symmetry group
        """
        norm_factor = {}
        for name, info in clusters.items():
            occ_count = self.occ_count_all[info["symm_group"]][name]
            norm_factor[name] = self._norm_factor(occ_count, info)
        return norm_factor

    def _occurence_count_all_symm_groups(self):
        occ_count_all = []
        for item in self.cluster_info:
            occ_count = {}
            for name, info in item.items():
                occ_count[name] = self._occurence_count(info)
            occ_count_all.append(occ_count)
        return occ_count_all

    def _occurence_count(self, cluster):
        """Count the number of occurences of each sub-cluster in the cluster

        Arguments:
        =========
        cluster: dict
            A dictionary with info about a particular cluster
        """
        occ_count = {}
        for indices, order in zip(cluster["indices"], cluster["order"]):
            key = self.index_key(cluster["ref_indx"], indices, order,
                                 cluster["equiv_sites"])
            occ_count[key] = occ_count.get(key, 0) + 1
        return occ_count

    def _norm_factor(self, occ_count, cluster):
        norm_count = {}
        for k, v in occ_count.items():
            tot_num = self._total_number_of_occurences(k, cluster["name"])
            num_unique = len(set(k.split("-")))
            norm_count[k] = float(tot_num)/(num_unique*v)
        return norm_count

    def _corresponding_subcluster(self, new_ref_indx, target_cluster, name):
        """Find the corresponding cluster when another index is ref_index."""
        cluster = self.cluster_info[self.symm_group[new_ref_indx]][name]
        for sub, order in zip(cluster["indices"], cluster["order"]):
            indices = [self.trans_matrix[new_ref_indx][indx] for indx in sub]
            indices = [new_ref_indx] + indices
            indices = [indices[j] for j in order]
            indices = self._order_equiv_sites(cluster["equiv_sites"], indices)

            if np.allclose(indices, target_cluster):
                subcluster = [cluster["ref_indx"]] + sub
                subcluster = [subcluster[i] for i in order]
                return self._order_equiv_sites(cluster["equiv_sites"],
                                               subcluster)

        raise RuntimeError("There are no matching subcluster. "
                           "This should never happen and is a bug.")

    def _total_number_of_occurences(self, key, name):
        """Get the total number of occurences."""
        indices = list(map(int, key.split("-")))
        tot_num = 0
        for ref_indx in set(indices):
            corr_cluster = \
                self._corresponding_subcluster(ref_indx, indices, name)
            new_key = list2str(corr_cluster)
            tot_num += \
                self.occ_count_all[self.symm_group[ref_indx]][name][new_key]
        return tot_num

    def show(self):
        """Return a string represenatation."""
        print(self._norm_factors)

    def _order_equiv_sites(self, equiv_sites, ordered_indices):
        """After the indices are ordered, adopt a consistent scheme
           within the equivalent sites."""
        for eq_group in equiv_sites:
            equiv_indices = [ordered_indices[i] for i in eq_group]
            equiv_indices.sort()
            for count, i in enumerate(eq_group):
                ordered_indices[i] = equiv_indices[count]
        return ordered_indices


@jit(nopython=True)
def cf_change_by_indx_jit(atoms_npy, ref_indx, new_symbid, cluster_indices,
                          equiv_deco, dup_factors, sp_mat, bf_npy):
        """Calculate the change in correlation function based on atomic index.

        This method tracks changes in correaltion function due to change in
        element type for atom with index = ref_indx. Passed trans_list refers
        to the indices of atoms that constitute the cluster
        (after translation).
        """
        symbol = atoms_npy[ref_indx]
        delta_cf = 0.0
        for dec_num in range(equiv_deco.shape[0]):
            dec = equiv_deco[dec_num, :]

            for subclust in range(cluster_indices.shape[0]):
                cf_new = 1.0
                cf_ref = 1.0
                for i in range(cluster_indices.shape[1]):
                    trans_indx = sp_mat.get(ref_indx,
                                            cluster_indices[subclust, i])

                    if trans_indx == ref_indx:
                        cf_new *= bf_npy[dec[i], new_symbid]
                        cf_ref *= bf_npy[dec[i], symbol]
                    else:
                        cf_new *= bf_npy[dec[i], atoms_npy[trans_indx]]
                        cf_ref *= bf_npy[dec[i], atoms_npy[trans_indx]]

                # If self interactions we need to down scale the contribution
                # because this cluster is present fewer times.
                # Example: If indices = [0, 23, 42], this particular clusters
                # exists when 0, 23 and 42 is reference index (total 3 times)
                # if the cluster is [0, 23, 23], this cluster exists only
                # when 0 and 23 is reference index (only 2 times).
                # Hence, we need to down scale the contribution by a factor
                # 2/3.
                scale = len(np.unique(cluster_indices[subclust, :])) / \
                    float(cluster_indices.shape[1])

                # We only inspect the effect of the change at one reference
                # index. However, we know that if we looped over all reference
                # indices one given clusters would appear exactly m times, if
                # m is the multiplicity factor. On the other hand when
                # inspecting only one cluster, that may not be the case.
                # Example: If indices = [[0, 1, 1], [0, 1, 1], [1, 0, 0]]
                # the situation is as follows. When 0 is reference index
                # [0, 1, 1] occures to times and [1, 0, 0] occures one time
                # When 1 is reference index [0, 1, 1] will occure one time
                # and [1, 0, 0] will occure two times. Since we only
                # inspect one of the cases, we need to correct for the
                # fact that for one particular reference index the number
                # of occurences can be distributed unevenly.
                scale *= dup_factors[subclust]
                delta_cf += scale*(cf_new - cf_ref)/equiv_deco.shape[0]
        return delta_cf
