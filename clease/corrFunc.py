"""Module for calculating correlation functions."""
from __future__ import print_function
import numpy as np
from ase.atoms import Atoms
from clease import CEBulk, CECrystal
from clease.tools import wrap_and_sort_by_position, equivalent_deco
from clease.tools import symbols2integer, bf2npyarray
from ase.db import connect
import multiprocessing as mp
from clease.jit import jit
from clease_cxx import PyCEUpdater

# workers can not be a member of CorrFunction since CorrFunctions is passed
# as argument to the map function. Hence, we leave it as a global variable,
# but it is initialized wheh the CorrFunction object is initialized.
workers = None


class ClusterNotTrackedError(Exception):
    pass


class CorrFunction(object):
    """Calculate the correlation function.

    Parameters:

    setting: settings object

    parallel: bool (optional)
        specify whether or not to use the parallel processing for `get_cf`
        method.

    num_core: int or "all" (optional)
        specify the number of cores to use for parallelization.
    """

    def __init__(self, setting, parallel=False, num_core="all"):
        if not isinstance(setting, (CEBulk, CECrystal)):
            raise TypeError("setting must be CEBulk or CECrystal "
                            "object")
        self.setting = setting

        self.parallel = parallel
        self.num_core = num_core

        bf = self.setting.basis_functions
        self.symb_id = symbols2integer(bf)

        self.bf_npy = bf2npyarray(bf, self.symb_id)
        self.tm = self._full_trans_matrix()
        self.atoms_npy = None

        if parallel:
            global workers
            if self.num_core == "all":
                num_proc = int(mp.cpu_count()/2)
            else:
                num_proc = int(self.num_core)
            if workers is None:
                workers = mp.Pool(num_proc)

    def _atoms2npy(self, atoms):
        array = np.zeros(len(atoms), dtype=np.int32)
        for atom in atoms:
            array[atom.index] = self.symb_id.get(atom.symbol, -1)
        return array

    def _full_trans_matrix(self):
        num_atoms = len(self.setting.trans_matrix)
        tm = np.zeros((num_atoms, num_atoms), dtype=np.int32)
        for i, row in enumerate(self.setting.trans_matrix):
            for k, v in row.items():
                tm[i, k] = v
        return tm

    def get_c1(self, atoms, dec):
        """Get correlation function for single-body clusters."""
        c1 = 0
        for element, spin in self.setting.basis_functions[dec].items():
            num_element = len([a for a in atoms if a.symbol == element])
            c1 += num_element * spin
        c1 /= float(len(atoms))
        return c1

    def _prepare_corr_func_calculation(self, atoms):
        if isinstance(atoms, Atoms):
            self.check_cell_size(atoms)
        else:
            raise TypeError('atoms must be Atoms object')

        self.atoms_npy = self._atoms2npy(atoms)
        self.tm = self._full_trans_matrix()

    def get_cf(self, atoms):
        """
        Calculate correlation functions for all possible clusters and return
        them in a dictionary format.

        Parameters:

        atoms: Atoms object
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('atoms must be an Atoms object')

        bf_list = list(range(len(self.setting.basis_functions)))
        cf = {}
        # ----------------------------------------------------
        # Compute correlation function up the max_cluster_size
        # ----------------------------------------------------
        # loop though all cluster sizes
        cf['c0'] = float(1.0)

        # Update singlets
        for dec in bf_list:
            cf['c1_{}'.format(dec)] = float(self.get_c1(atoms, dec))

        cnames = self.setting.cluster_names
        if self.parallel:
            # Pre-calculate nessecary stuff prior to parallel execution
            self._prepare_corr_func_calculation(atoms)

            args = [(self, atoms, name) for name in cnames]
            res = workers.map(get_cf_parallel, args)
            cf = {}
            for r in res:
                cf.update(r)
            return cf
        return self.get_cf_by_cluster_names(atoms, cnames)

    def get_cf_by_cluster_names(self, atoms, cluster_names, warm=False):
        """
        Calculate correlation functions of the specified clusters and return
        them in a dictionary format.

        Parameters:

        atoms: Atoms object

        cluster_names: list
            names (str) of the clusters for which the correlation functions are
            calculated for the structure provided in atoms
        """

        self._confirm_cluster_names_exists(cluster_names)

        if not self.parallel:
            self._prepare_corr_func_calculation(atoms)

        eci = {name: 1.0 for name in cluster_names}
        cf = {name: 1.0 for name in cluster_names}
        updater = PyCEUpdater(atoms, self.setting, cf, eci,
                              self.setting.cluster_info)
        cf = updater.calculate_cf_from_scratch()
        return cf

        cf = {}
        for name in cluster_names:
            if name == 'c0':
                cf[name] = float(1.0)
                continue
            prefix = name.rpartition('_')[0]
            dec = name.rpartition('_')[-1]
            dec_list = [int(i) for i in dec]
            # find c{num} in cluster type
            n = int(prefix[1])

            if n == 1:
                cf[name] = float(self.get_c1(atoms, int(dec)))
                continue

            sp = 0.
            count = 0
            # loop through the symmetry inequivalent groups
            # for symm in range(self.num_trans_symm):
            for cluster_set in self.setting.cluster_info:
                if prefix not in cluster_set.keys():
                    continue
                cluster = cluster_set[prefix]
                sp_temp, count_temp = \
                    self._spin_product(self.atoms_npy, cluster, dec_list)
                sp += sp_temp
                count += count_temp
            cf_temp = sp / count
            cf['{}_{}'.format(prefix, dec)] = float(cf_temp)

        return cf

    def reconfigure_db_entries(self, select_cond=None, verbose=True):
        """Reconfigure the correlation function values of the entries in DB.

        Parameters:

        select_cond: list
            -None (default): select every item in DB with
                             "struct_type='initial'"
            -else: select based on the condictions provided
                  (struct_type='initial' is not automatically included)

        verbose: bool
            print the progress of reconfiguration if set to *True*
        """
        db = connect(self.setting.db_name)
        select = []
        if select_cond is not None:
            for cond in select_cond:
                select.append(cond)
        else:
            select = [('struct_type', '=', 'initial')]

        # get how many entries need to be reconfigured
        row_ids = [row.id for row in db.select(select)]
        num_reconf = len(row_ids)
        if verbose:
            print('{} entries will be reconfigured'.format(num_reconf))

        import sqlite3
        for count, row_id in enumerate(row_ids):
            # TODO: Should this be part of DB API?
            tab_name = "{}_cf".format(self.setting.bf_scheme.name)
            con = sqlite3.connect(self.setting.db_name, timeout=600)
            cur = con.cursor()
            try:
                cur.execute("DELETE FROM {} WHERE ID=?".format(tab_name),
                            (row_id,))
                con.commit()
            except sqlite3.OperationalError as exc:
                print(str(exc) + ". did not delete anything.")
            con.close()

            # get new CF based on setting
            if verbose:
                print("updating {} of {} entries".format(count+1, num_reconf),
                      end="\r")
            atoms = wrap_and_sort_by_position(db.get(id=row_id).toatoms())
            cf = self.get_cf(atoms)
            db.update(row_id, external_tables={tab_name: cf})

        if verbose:
            print("reconfiguration completed")

    def reconfigure_inconsistent_cf_table_entries(self):
        """Find and correct inconsistent correlation functions in table."""
        inconsistent_ids = self.check_consistency_of_cf_table_entries()

        if len(inconsistent_ids) == 0:
            return True

        for count, id in enumerate(inconsistent_ids):
            print("updating {} of {} entries"
                  "".format(count+1, len(inconsistent_ids)), end="\r")
            self.reconfigure_db_entries(select_cond=[('id', '=', id)],
                                        verbose=False)
        print("reconfiguration completed")

    def check_consistency_of_cf_table_entries(self):
        """Get IDs of the structures with inconsistent correlation functions.

        Note: consisent structures have the exactly the same list of cluster
              names as stored in setting.cluster_names.
        """
        db = connect(self.setting.db_name)
        tab_name = "{}_cf".format(self.setting.bf_scheme.name)
        cnames = sorted(self.setting.cluster_names.copy())
        inconsistent_ids = []
        for row in db.select('struct_type=initial'):
            tab_entries = row.get(tab_name, {})
            row_cnames = sorted(list(tab_entries.keys()))
            if row_cnames != cnames:
                inconsistent_ids.append(row.id)

        if len(inconsistent_ids) > 0:
            print("{} inconsistent entries found in '{}' table."
                  "".format(len(inconsistent_ids), tab_name))
            for id in inconsistent_ids:
                print('  id: {}, name: {}'.format(id, db.get(id).name))
        else:
            print("'{}' table has no inconsistent entries.".format(tab_name))
        return inconsistent_ids

    def _spin_product(self, atoms, cluster, deco):
        """Get spin product of a given cluster.

        Parameters:

        atoms: Atoms object

        cluster: dict
            A dictionary containing all necessary information about the
            family of cluster (i.e., list of indices, order, equivalent sites,
            symmetry group, etc.).

        deco: tuple
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom.
        """
        sp = 0.
        count = 0

        # spin product of each atom in the symmetry equivalent group
        indices_of_symm_group = \
            self.setting.index_by_trans_symm[cluster["symm_group"]]
        ref_indx_grp = indices_of_symm_group[0]

        eq_sites = list(cluster["equiv_sites"])
        equiv_deco = np.array(equivalent_deco(deco, eq_sites))

        # Convert to numpy arrays
        indices = np.array(cluster["indices"])
        order = np.array(cluster["order"])

        for ref_indx in indices_of_symm_group:
            sp_temp, count_temp = _sp_same_shape_deco_for_ref_indx_jit(
                atoms, ref_indx, indices, order, ref_indx_grp, equiv_deco,
                self.tm, self.bf_npy)
            sp += sp_temp
            count += count_temp
        return sp, count

    def check_cell_size(self, atoms):
        """Check the size of provided cell and create a template if necessary.

        Parameters:

        atoms: Atoms object
            Unrelaxed structure
        """
        self.setting.set_active_template(atoms=atoms, generate_template=True)
        return atoms

    def _cluster_name_exists(self, cluster_name):
        """Return True if cluster name exists. Otherwise False.

        Parameters:

        cluster_name: str
            Cluster name to check
        """
        return cluster_name in self.setting.cluster_names

    def _confirm_cluster_names_exists(self, cluster_names):
        if not set(cluster_names).issubset(self.setting.cluster_names):
            raise ClusterNotTrackedError(
                "The correlation function of non-existing cluster is "
                "requested, but the name does not exist in "
                "ClusterExpansionSetting. Check that the cutoffs are "
                "correct, and try to run reconfigure_settings")


def get_cf_parallel(args):
    cf = args[0]
    atoms = args[1]
    name = args[2]
    return cf.get_cf_by_cluster_names(atoms, [name])


@jit(nopython=True)
def _spin_product_one_cluster_jit(atoms, ref_indx, cluster_indices,
                                  order, ref_indx_grp, equiv_deco,
                                  trans_matrix, bf):
    """Compute sp of cluster with same shape and deco for given ref atom.

    Parameters:

    atoms: np.ndarray
        1D numpy array representation of the atoms object. Each symbol
        has a unique number set by the CorrFunction class. Example: If
        curr_func.symb_id = {"Al": 0, "Cu": 1, "Li": 2} and the symbols
        are ["Al", "Al", "Li", "Cu", "Li"], this array would be
        [0, 0, 2, 1, 2]

    ref_indx: int
        Index of the atom used as a reference to get clusters.

    cluster_indices: np.ndarray
        1D numpy array of inidices that constitue a cluster. Each row
        correspond to a sub-cluster. For a triplet this could be
        [0, 1]
        hence the reference index itself is not included.

    indx_order: np.ndarray
        A 1D array of how the indices in "cluster_indices" should be
        ordered. The indices of atoms are sorted in a decrease order of
        internal distances to other members of the cluster.

    ref_indx_grp: int
        Index of the reference atom used for the translational symmetry
        group.

    deco: np.ndarray
        Decoration number that specifies which basis function should be
        used for getting the spin variable of each atom. Each row in the
        array represents the different combination that by symmetry should
        be averaged.
    trans_matrix: np.ndarray
        2D Numpy array of the full translation matrix.
    bf: np.ndarray
        2D Numpy array of holding the basis functions.
    """
    count = 0
    sp = 0.0
    indices = np.zeros(len(cluster_indices) + 1)
    indices[0] = ref_indx_grp
    for i in range(len(cluster_indices)):
        indices[i+1] = cluster_indices[i]

    sorted_indices = np.zeros(len(indices), dtype=np.int32)
    for i in range(len(indices)):
        sorted_indices[i] = indices[order[i]]

    for dec_num in range(equiv_deco.shape[0]):
        dec = equiv_deco[dec_num, :]
        sp_temp = 1.0
        # loop through indices of atoms in each cluster
        for i in range(len(sorted_indices)):
            indx = sorted_indices[i]
            trans_indx = trans_matrix[ref_indx, indx]
            sp_temp *= bf[dec[i], atoms[trans_indx]]
        sp += sp_temp
        count += 1
    num_equiv = float(equiv_deco.shape[0])
    return sp/num_equiv, count/num_equiv


@jit(nopython=True)
def _sp_same_shape_deco_for_ref_indx_jit(atoms, ref_indx, indices, order,
                                         ref_indx_grp, equiv_deco, tm, bf):
    """Compute sp of cluster with same shape and deco for given ref atom.

    Parameters:

    atoms: np.ndarray
        1D numpy array representation of the atoms object. Each symbol
        has a unique number set by the CorrFunction class. Example: If
        curr_func.symb_id = {"Al": 0, "Cu": 1, "Li": 2} and the symbols
        are ["Al", "Al", "Li", "Cu", "Li"], this array would be
        [0, 0, 2, 1, 2]

    ref_indx: int
        Index of the atom used as a reference to get clusters.

    indx_list: np.ndarray
        2D numpy array of inidices that constitue a cluster. Each row
        correspond to a sub-cluster. For a triplet this could be
        [[0, 1]
            [5, 2]
            [7, 8]]
            hence the reference index itself is not included.

    indx_order: np.ndarray
        A 2D array of how the indices in "indx_list" should be ordered.
        The indices of atoms are sorted in a decrease order of internal
        distances to other members of the cluster.

    ref_indx_grp: int
        Index of the reference atom used for the translational symmetry
        group.

    deco: np.ndarray
        Decoration number that specifies which basis function should be
        used for getting the spin variable of each atom. Each row in the
        array represents the different combination that by symmetry should
        be averaged.

    tm: np.ndarray
        2D Numpy array of the full translation matrix.

    bf: np.ndarray
        2D Numpy array of holding the basis functions.
    """
    count = 0
    sp = 0.0
    for i in range(indices.shape[0]):
        temp_sp, temp_cnt = \
            _spin_product_one_cluster_jit(atoms, ref_indx,
                                          indices[i, :], order[i, :],
                                          ref_indx_grp, equiv_deco, tm, bf)
        sp += temp_sp
        count += temp_cnt
    return sp, count
