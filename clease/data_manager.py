from clease import _logger
from ase.db import connect
import numpy as np
from clease.tools import add_file_extension
from ase.db.core import parse_selection
from typing import Tuple, List, Dict, Set
import sqlite3


class InconsistentDataError(Exception):
    pass


class DataManager(object):
    """
    DataManager is a class for extracting data from CLEASE databases to be
    used to fit ECIs

    Parameters:

    db_name:
        Name of the database
    """
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._X = None
        self._y = None
        self._feat_names = None
        self._target_name = None

    def get_data(self, select_cond: List[tuple], feature_getter: callable,
                 target_getter: callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the design matrix X and the target data y

        Parameters:

        select_cond: list
            List of tuples with selection conditions
            (e.g. [('converged', '='' True)])

        feature_getter:
            Callable object that returns design matrix. It
            also needs a method get_feature_names that returns a name of
            each feature. The passed instance should take a list of IDs
            as input argument and return a numpy array corresponding to
            the design matrix

        target_getter:
            Callable object that extracts the target value. The __call__
            method takes a list of ids in the database and return a numpy
            array with the target values (e.g. energy per atom)

        Example:

        >>> from clease.data_manager import (
        ... CorrelationFunctionGetter, FinalStructEnergyGetter,
        ... DataManager)
        >>> from ase.db import connect
        >>> from ase import Atoms
        >>> from ase.calculators.emt import EMT
        >>> db_name = 'somedb.db'
        >>> db = connect(db_name)  # Make sure that the DB exists
        >>> db.write(Atoms('Cu'), external_tables={'polynomial_cf': {'c0': 1.0}}, final_struct_id=2, converged=True)
        1
        >>> final = Atoms('Cu')
        >>> final.set_calculator(EMT())
        >>> _ = final.get_potential_energy()
        >>> db.write(final)
        2
        >>> feat_getter = CorrelationFunctionGetter(db_name, ['c0', 'c1_0',
        ... 'c2_d0000_0_00'], 'polynomial_cf')
        >>> targ_getter = FinalStructEnergyGetter(db_name)
        >>> manager = DataManager(db_name)
        >>> X, y = manager.get_data([('converged', '=', 1)], feat_getter, targ_getter)
        """
        keys, cmps = parse_selection(select_cond)
        db = connect(self.db_name)
        sql, args = db.create_select_statement(keys, cmps)

        # Extract the ids in the database that corresponds to select_cond
        sql = sql.replace('systems.*', 'systems.id')
        with connect(self.db_name) as db:
            con = db.connection
            cur = con.cursor()
            cur.execute(sql, args)
            ids = [row[0] for row in cur.fetchall()]
        ids.sort()

        # Extract design matrix and the target values
        cfm = feature_getter(ids)
        target = target_getter(ids)

        self._X = np.array(cfm, dtype=float)
        self._y = np.array(target)
        self._feat_names = feature_getter.names()
        self._target_name = target_getter.name

        nrows = self._X.shape[0]
        num_data = len(self._y)
        if nrows != num_data:
            raise InconsistentDataError(
                f"Num. rows in deisgn matrix {nrows}."
                f"Num. data points {num_data}\n"
                "This is normally caused by a non-converged structure "
                "being extracted from database. Check that all structures"
                "extracted from DB using your query has a corresponding"
                "final structure.")
        return self._X, self._y

    def to_csv(self, fname: str):
        """
        Export the dataset used to fit a model y = Xc where y is typically the
        DFT energy per atom and c is the unknown ECIs. This function exports
        the data to a csv file with the following format

        # ECIname_1, ECIname_2, ..., ECIname_n, E_DFT
        0.1, 0.4, ..., -0.6, -2.0
        0.3, 0.2, ..., -0.9, -2.3

        thus each row in the file contains the correlation function values and
        the corresponding DFT energy value.

        Parameter:

        fname: str
            Filename to write to. Typically this should end with .csv
        """
        if self._X is None:
            return
        fname = add_file_extension(fname, 'csv')
        header = f",".join(self._feat_names) + f",{self._target_name}"
        data = np.hstack((self._X,
                          np.reshape(self._y, (len(self._y), -1))))
        np.savetxt(fname, data, delimiter=",", header=header)
        _logger(f"Dataset exported to {fname}")

    def get_matching_names(self, pattern):
        """
        Get names that matches pattern

        Parameters:

        pattern: str
            Pattern which the string should contain.

        Example:

        If the names are ['abc', 'def', 'gbcr'] and the passed pattern
        is 'bc', then ['abc', 'gbcr'] will be returned
        """
        return [n for n in self._feat_names if pattern in n]

    def get_cols(self, names):
        """
        Get all columns corresponding to the names

        Parameters:

        names: list
            List of names (e.g. ['c0', 'c1_1'])
        """
        name_idx = {n: i for i, n in enumerate(self._feat_names)}
        indices = [name_idx[n] for n in names]
        return self._X[:, indices]


class CorrFuncEnergyDataManager(DataManager):
    """
    CorrFuncFinalEnergyDataManager is a convenience class provided
    to handle the standard case where the features are correlation functions
    and the target is the DFT energy per atom

    Parameters

    db_name: Name of the database being passed

    cf_names: List with the correlation function names to extract

    tab_name: Name of the table where the correlation functions are stored
    """
    def __init__(self, db_name: str, cf_names: List[str], tab_name: str):
        DataManager.__init__(self, db_name)
        self.tab_name = tab_name
        self.cf_names = cf_names

    def get_data(self, select_cond):
        """
        Return X and y, where X is the design matrix containing correlation
        functions and y is the DFT energy per atom.

        Parameters:

        select_cond: list
            List with select conditions for the database
            (e.g. [('converged', '=', True)])
        """
        return DataManager.get_data(
            self, select_cond,
            CorrelationFunctionGetter(self.db_name, self.cf_names, self.tab_name),
            FinalStructEnergyGetter(self.db_name))


class CorrelationFunctionGetter(object):
    """
    CorrelationFunctionGetter is a class that extracts
    the correlation functions from an AtomsRow object

    Parameters:

    db_name:
        Name of the database
    cf_names: list
        List with the names of the correlation functions

    tab_name: str
        Name of the external table where the correlation functions
        are stored
    """
    def __init__(self, db_name: str, cf_names: List[str], tab_name: str):
        self.db_name = db_name
        self.cf_names = cf_names
        self.tab_name = tab_name

    def names(self):
        """
        Return a name of each column
        """
        return self.cf_names

    def __call__(self, ids: List[int]) -> np.ndarray:
        sql = f"SELECT * FROM {self.tab_name}"
        id_cf_values = {}
        id_cf_names = {}

        id_set = set(ids)

        with connect(self.db_name) as db:
            con = db.connection
            cur = con.cursor()
            cur.execute(sql)

            # Extract the correlation function name and value. The ID is also
            # extracted to check if it is in ids. The correlation functions
            # are placed in a dictionary because the IDs is not nessecarily
            # a monotone sequence of ints
            for name, value, db_id in cur.fetchall():
                if db_id in id_set:
                    cur_row = id_cf_values.get(db_id, [])
                    names = id_cf_names.get(db_id, [])

                    cur_row.append(value)
                    names.append(name)

                    id_cf_values[db_id] = cur_row
                    id_cf_names[db_id] = names

        # Make sure that all the rows are ordered in the same way
        cf_name_col = {n: i for i, n in enumerate(self.cf_names)}

        cf_matrix = np.zeros((len(id_cf_values), len(cf_name_col)))
        row = 0

        # Convert the correlation function dictionary to a numpy 2D array
        # such that is can be passed to a fitting algorithm
        for row, db_id in enumerate(ids):
            cf_values = id_cf_values[db_id]
            cf_names = id_cf_names[db_id]
            cols = [cf_name_col[name] for name in cf_names]
            cf_matrix[row, :] = np.array(cf_values)[cols]
            row += 1
        return cf_matrix


class FinalStructEnergyGetter(object):
    """
    FinalStructEnergyGetter is a callable class that returns the final energy
    (typically after structure relaxation) corresponding to the passed
    AtomsRow object.
    """
    def __init__(self, db_name: str):
        self.db_name = db_name

    @property
    def name(self):
        return "E_DFT (eV/atom)"

    def __call__(self, ids: List[int]) -> np.ndarray:
        sql = f"SELECT value, id FROM number_key_values WHERE key='final_struct_id'"
        id_set = set(ids)
        # Map beetween the initial structure and the index in the id list
        init_struct_idx = {idx: i for i, idx in enumerate(ids)}

        with connect(self.db_name) as db:
            con = db.connection
            cur = con.cursor()
            cur.execute(sql)
            final_struct_ids = []

            # Map the between final struct id and initial struct id
            init_struct_ids = {}
            for final_id, init_id in cur.fetchall():
                if init_id in id_set:
                    final_struct_ids.append(final_id)
                    init_struct_ids[final_id] = init_id

            # Extract the number of atoms of the initial structure
            num_atoms = extract_num_atoms(cur, id_set)

            final_struct_ids = set(final_struct_ids)
            sql = f"SELECT id, energy FROM systems"
            cur.execute(sql)
            energies = np.zeros(len(final_struct_ids))

            # Populate the energies array
            for final_id, energy in cur.fetchall():
                if final_id in final_struct_ids:
                    init_id = init_struct_ids[final_id]
                    idx = init_struct_idx[init_id]
                    energies[idx] = energy/num_atoms[init_id]
        return energies


class FinalVolumeGetter(object):
    """
    Class that extracts the final volume per atom of the relaxed structure
    """
    def __init__(self, db_name: str):
        self.db_name = db_name

    @property
    def name(self):
        return "Volume (A^3)"

    def __call__(self, ids: List[int]) -> np.ndarray:
        id_set = set(ids)
        query = "SELECT value, id FROM number_key_values WHERE key='final_struct_id'"

        init_ids = {}
        id_idx = {db_id: i for i, db_id in enumerate(ids)}
        with connect(self.db_name) as db:
            cur = db.connection.cursor()
            cur.execute(query)
            final_struct_ids = []

            # Extract the final struct id. Create a mapping between
            # final structure ids and initial structure ids for later
            # reference
            for value, db_id in cur.fetchall():
                final_id = int(value)
                if db_id in id_set:
                    final_struct_ids.append(final_id)
                    init_ids[final_id] = db_id

            num_atoms = extract_num_atoms(cur, id_set)

            final_struct_ids = set(final_struct_ids)
            query = 'SELECT id, volume FROM systems'
            cur.execute(query)
            volumes = np.zeros(len(init_ids))

            # Extract the volume of the final structure and normalize it by
            # the number of atoms in the initial structure. The reason why
            # the number of atoms in the initial structure needs to be used,
            # is that vacancies are typically removed before the DFT run
            # starts. Consequently, they are missing in the final structure
            # and the number of atoms in that structure will be wrong
            for db_id, vol in cur.fetchall():
                if db_id in final_struct_ids:
                    init_id = init_ids[db_id]
                    idx = id_idx[init_id]
                    volumes[idx] = vol/num_atoms[init_id]
        return volumes


class CorrFuncVolumeDataManager(DataManager):
    """
    CorrFuncVolumeDataManager is a convenience class provided
    to handle the standard case where the features are correlation functions
    and the target is the volume of the relaxed cell

    Parameters

    db_name: Name of the database being passed

    cf_names: List with the correlation function names to extract

    tab_name: Name of the table where the correlation functions are stored
    """
    def __init__(self, db_name: str, cf_names: List[str], tab_name: str):
        DataManager.__init__(self, db_name)
        self.tab_name = tab_name
        self.cf_names = cf_names

    def get_data(self, select_cond: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return X and y, where X is the design matrix containing correlation
        functions and y is the DFT energy per atom.

        Parameters:

        select_cond: list
            List with select conditions for the database
            (e.g. [('converged', '=', True)])
        """
        return DataManager.get_data(
            self, select_cond,
            CorrelationFunctionGetter(self.db_name, self.cf_names,
                                      self.tab_name), FinalVolumeGetter(self.db_name))


def extract_num_atoms(cur: sqlite3.Cursor,
                      ids: Set[int]) -> Dict[int, int]:
    """
    Extract the number of atoms for all ids

    cur: SQL-cursor object
    ids: Set with IDs in the database.
    """
    sql = "SELECT id, natoms FROM systems"
    cur.execute(sql)
    num_atoms = {}
    for db_id, natoms in cur.fetchall():
        if db_id in ids:
            num_atoms[db_id] = natoms
    return num_atoms
