"""Module for calculating correlation functions."""
from __future__ import print_function
from ase.atoms import Atoms
from clease import ClusterExpansionSettings
from clease.tools import wrap_and_sort_by_position
from ase.db import connect
from clease_cxx import PyCEUpdater
from clease import _logger


class ClusterNotTrackedError(Exception):
    pass


class CorrFunction(object):
    """Calculate the correlation function.

    Parameters:

    settings: settings object

    parallel: bool (optional)
        specify whether or not to use the parallel processing for `get_cf`
        method.

    num_core: int or "all" (optional)
        specify the number of cores to use for parallelization.
    """

    def __init__(self, settings, parallel=False, num_core="all"):
        if not isinstance(settings, ClusterExpansionSettings):
            raise TypeError("setting must be CEBulk or CECrystal "
                            "object")
        self.settings = settings

    def get_cf(self, atoms):
        """
        Calculate correlation functions for all possible clusters and return
        them in a dictionary format.

        Parameters:

        atoms: Atoms object
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('atoms must be an Atoms object')
        cf_names = self.settings.all_cf_names
        return self.get_cf_by_names(atoms, cf_names)

    def get_cf_by_names(self, atoms, cf_names, warm=False):
        """
        Calculate correlation functions of the specified clusters and return
        them in a dictionary format.

        Parameters:

        atoms: Atoms object

        cf_names: list
            names of correlation functions that will be calculated for
            the structure provided in atoms
        """

        if isinstance(atoms, Atoms):
            self.check_cell_size(atoms)
        else:
            raise TypeError('atoms must be Atoms object')

        self._confirm_cf_names_exists(cf_names)

        eci = {name: 1.0 for name in cf_names}
        cf = {name: 1.0 for name in cf_names}
        updater = PyCEUpdater(atoms, self.settings, cf, eci,
                              self.settings.cluster_list)
        cf = updater.calculate_cf_from_scratch(atoms, cf_names)
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
        tab_name = "{}_cf".format(self.settings.basis_func_type.name)
        db = connect(self.settings.db_name)
        db.delete_external_table(tab_name)
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
            _logger(f"{num_reconf} entries will be reconfigured")
        for count, row_id in enumerate(row_ids):

            if verbose:
                _logger(f"updating {count+1} of {num_reconf} entries",
                        end="\r")
            atoms = wrap_and_sort_by_position(db.get(id=row_id).toatoms())
            cf = self.get_cf(atoms)
            db.update(row_id, external_tables={tab_name: cf})

        if verbose:
            _logger("\nreconfiguration completed")

    def reconfigure_inconsistent_cf_table_entries(self):
        """Find and correct inconsistent correlation functions in table."""
        inconsistent_ids = self.check_consistency_of_cf_table_entries()

        if len(inconsistent_ids) == 0:
            return True

        for count, id in enumerate(inconsistent_ids):
            _logger(f"updating {count+1} of {len(inconsistent_ids)} entries",
                    end="\r")
            self.reconfigure_db_entries(select_cond=[('id', '=', id)],
                                        verbose=False)
        _logger("\nreconfiguration completed")

    def check_consistency_of_cf_table_entries(self):
        """Get IDs of the structures with inconsistent correlation functions.

        Note: consisent structures have the exactly the same list of cluster
              names as stored in settings.cf_names.
        """
        db = connect(self.settings.db_name)
        tab_name = f"{self.settings.basis_func_type.name}_cf"
        cf_names = sorted(self.settings.all_cf_names)
        inconsistent_ids = []
        for row in db.select('struct_type=initial'):
            tab_entries = row.get(tab_name, {})
            row_cnames = sorted(list(tab_entries.keys()))
            if row_cnames != cf_names:
                inconsistent_ids.append(row.id)

        if len(inconsistent_ids) > 0:
            _logger(f"{len(inconsistent_ids)} inconsistent entries found in "
                    f"'{tab_name}' table.")
            for id in inconsistent_ids:
                _logger(f"  id: {id}, name: {db.get(id).name}")
        else:
            _logger(f"'{tab_name}' table has no inconsistent entries.")
        return inconsistent_ids

    def check_cell_size(self, atoms):
        """Check the size of provided cell and create a template if necessary.

        Parameters:

        atoms: Atoms object
            Unrelaxed structure
        """
        self.settings.set_active_template(atoms=atoms)
        return atoms

    def _cf_name_exists(self, cf_name):
        """Return True if cluster name exists. Otherwise False.

        Parameters:

        cluster_name: str
            Cluster name to check
        """
        return cf_name in self.settings.all_cf_names

    def _confirm_cf_names_exists(self, cf_names):
        if not set(cf_names).issubset(self.settings.all_cf_names):
            raise ClusterNotTrackedError(
                "The correlation function of non-existing cluster is "
                "requested, but the name does not exist in "
                "ClusterExpansionSettings. Check that the cutoffs are "
                "correct, and try to run reconfigure_settings")
