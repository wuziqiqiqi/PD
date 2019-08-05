"""Module for calculating correlation functions."""
from __future__ import print_function
import numpy as np
from ase.atoms import Atoms
from clease import CEBulk, CECrystal
from clease.tools import wrap_and_sort_by_position, equivalent_deco
from clease.tools import symbols2integer, bf2npyarray
from ase.db import connect
from clease_cxx import PyCEUpdater


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

    def get_cf(self, atoms):
        """
        Calculate correlation functions for all possible clusters and return
        them in a dictionary format.

        Parameters:

        atoms: Atoms object
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('atoms must be an Atoms object')
        cnames = self.setting.cluster_names
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

        if isinstance(atoms, Atoms):
            self.check_cell_size(atoms)
        else:
            raise TypeError('atoms must be Atoms object')

        self._confirm_cluster_names_exists(cluster_names)

        eci = {name: 1.0 for name in cluster_names}
        cf = {name: 1.0 for name in cluster_names}
        updater = PyCEUpdater(atoms, self.setting, cf, eci,
                              self.setting.cluster_info)
        cf = updater.calculate_cf_from_scratch(atoms, cluster_names)
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