"""Module for calculating correlation functions."""
from typing import Iterator, Tuple, Dict, Any
import logging
from ase.atoms import Atoms

from clease_cxx import PyCEUpdater
from .settings import ClusterExpansionSettings
from .tools import wrap_and_sort_by_position
from . import db_util

logger = logging.getLogger(__name__)
__all__ = ("CorrFunction", "ClusterNotTrackedError")

# Type alias for a Correlation function
CF_T = Dict[str, float]


class ClusterNotTrackedError(Exception):
    """A cluster is not being tracked"""


class CorrFunction:
    """Class for calculating the correlation functions.

    Parameters:

        settings (ClusterExpansionSettings): The settings object which defines the
            cluster expansion parameters.
    """

    def __init__(self, settings: ClusterExpansionSettings):
        self.settings = settings

    @property
    def settings(self) -> ClusterExpansionSettings:
        return self._settings

    @settings.setter
    def settings(self, value: Any) -> None:
        if not isinstance(value, ClusterExpansionSettings):
            raise TypeError(f"Setting must be a ClusterExpansionSettings object, got {value!r}")
        self._settings = value

    def connect(self, **kwargs):
        return self.settings.connect(**kwargs)

    def get_cf(self, atoms) -> CF_T:
        """
        Calculate correlation functions for all possible clusters and return
        them in a dictionary format.

        Parameters:

            atoms (Atoms): The atoms object
        """
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an Atoms object")
        cf_names = self.settings.all_cf_names
        return self.get_cf_by_names(atoms, cf_names)

    def get_cf_by_names(self, atoms, cf_names) -> CF_T:
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
            self.set_template(atoms)
        else:
            raise TypeError("atoms must be Atoms object")

        self._confirm_cf_names_exists(cf_names)

        eci = {name: 1.0 for name in cf_names}
        cf = {name: 1.0 for name in cf_names}
        updater = PyCEUpdater(atoms, self.settings, cf, eci, self.settings.cluster_list)
        cf = updater.calculate_cf_from_scratch(atoms, cf_names)
        return cf

    def reconfigure_single_db_entry(self, row_id: int) -> None:
        """Reconfigure a single DB entry. Assumes this is the initial structure,
        and will not check that.

        Parameters:

            row_id: int
                The ID of the row to be reconfigured.
        """
        with self.connect() as db:
            atoms = wrap_and_sort_by_position(db.get(id=row_id).toatoms())
            cf = self.get_cf(atoms)
            db_util.update_table(db, row_id, self.cf_table_name, cf)

    @property
    def cf_table_name(self) -> str:
        """Name of the table which holds the correlation functions."""
        return f"{self.settings.basis_func_type.name}_cf"

    def clear_cf_table(self) -> None:
        """Delete the external table which holds the correlation functions."""
        with self.connect() as db:
            db.delete_external_table(self.cf_table_name)

    def reconfigure_db_entries(self, select_cond=None, verbose=False):
        """Reconfigure the correlation function values of the entries in DB.

        Parameters:

            select_cond: One of either:

                - None (default): select every item in DB with ``struct_type='initial'``
                - Select based on the condictions provided (``struct_type='initial'`` is
                  not automatically included)

            verbose (bool):
                print the progress of reconfiguration if set to *True*
        """
        select = format_selection(select_cond)
        db = self.connect()

        # get how many entries need to be reconfigured
        num_reconf = db.count(select)
        msg = f"{num_reconf} entries will be reconfigured"
        logger.info(msg)
        if verbose:
            print(msg)

        for row_id, count, total in self.iter_reconfigure_db_entries(select_cond=select_cond):
            msg = f"Updated {count} of {total} entries. Current ID: {row_id}"
            if verbose:
                print(msg, end="\r")
            logger.debug(msg)

        if verbose:
            print("\nreconfiguration completed")
        logger.info("Reconfiguration complete")

    def iter_reconfigure_db_entries(self, select_cond=None) -> Iterator[Tuple[int, int, int]]:
        """Iterator which reconfigures the correlation function values in the DB,
        which yields after each reconfiguration and reports on the progress.

        For more information, see :py:meth:`~reconfigure_db_entries`.

        Yields:
            Tuple[int, int, int]: (row_id, count, total) A tuple containing the ID
            of the row which was just reconfigured, current
            count which has been reconfigured, as well as the total number of
            reconfigurations which will be performed.
            The percentage-wise progress is thus (count / total) * 100.
        """
        # Setup, ensure CF's are cleared first
        self.clear_cf_table()
        db = self.connect()
        select = format_selection(select_cond)
        # Fetch the total number of reconfigures
        total = db.count(select)
        for count, row in enumerate(db.select(select), start=1):
            row_id = row.id
            self.reconfigure_single_db_entry(row_id)
            # Yield the current progress
            yield row_id, count, total

    def reconfigure_inconsistent_cf_table_entries(self):
        """Find and correct inconsistent correlation functions in table."""
        inconsistent_ids = self.check_consistency_of_cf_table_entries()

        if len(inconsistent_ids) == 0:
            return

        logger.info("Reconfiguring correlation functions")
        for count, bad_id in enumerate(inconsistent_ids):
            logger.debug(
                "Updating %s of %s entries (id %s)",
                count + 1,
                len(inconsistent_ids),
                bad_id,
            )
            self.reconfigure_db_entries(select_cond=[("id", "=", bad_id)], verbose=False)
        logger.info("Reconfiguration completed")

    def check_consistency_of_cf_table_entries(self):
        """Get IDs of the structures with inconsistent correlation functions.

        Note: consisent structures have the exactly the same list of cluster
        names as stored in settings.cf_names.
        """
        db = self.connect()
        tab_name = self.cf_table_name
        cf_names = sorted(self.settings.all_cf_names)
        inconsistent_ids = []
        for row in db.select("struct_type=initial"):
            tab_entries = row.get(tab_name, {})
            row_cnames = sorted(list(tab_entries.keys()))
            if row_cnames != cf_names:
                inconsistent_ids.append(row.id)

        if len(inconsistent_ids) > 0:
            logger.warning(
                "%d inconsistent entries found in table %s",
                len(inconsistent_ids),
                tab_name,
            )
            for bad_id in inconsistent_ids:
                logger.warning("  id: %s, name: %s", bad_id, db.get(bad_id).name)
        else:
            logger.info("'%s' table has no inconsistent entries.", tab_name)
        return inconsistent_ids

    def set_template(self, atoms: Atoms) -> None:
        """Check the size of provided cell and set as the currently active
        template in the settings object.

        Parameters:

            atoms (Atoms):
                Unrelaxed structure
        """
        self.settings.set_active_template(atoms=atoms)

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
                "correct, and try to run reconfigure_settings"
            )


def format_selection(select_cond=None, default_struct_type="initial"):
    """DB selection formatter. Will default to selecting
    all initial structures if None is specified."""
    select = []
    if select_cond is not None:
        for cond in select_cond:
            select.append(cond)
    else:
        select = [("struct_type", "=", default_struct_type)]
    return select
