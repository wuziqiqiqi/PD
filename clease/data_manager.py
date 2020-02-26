from clease import _logger
from ase.db import connect
import numpy as np
from clease.tools import add_file_extension
from ase.db.core import parse_selection
from typing import Tuple, List


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
        >>> db.write(Atoms(), external_tables={'polynomial_cf': {'c0': 1.0}}, final_struct_id=2, converged=True)
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
        sql = sql.replace('systems.*', 'systems.id')
        with connect(self.db_name) as db:
            con = db.connection
            cur = con.cursor()
            cur.execute(sql, args)
            ids = [row[0] for row in cur.fetchall()]
        ids.sort()
        cfm = feature_getter(ids)
        target = target_getter(ids)

        self._X = np.array(cfm, dtype=float)
        self._y = np.array(target)
        self._feat_names = feature_getter.names()
        self._target_name = target_getter.name()

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
        id_seq = ','.join(['?']*len(ids))
        name_seq = ','.join(['?']*len(self.cf_names))
        sql = f"SELECT * FROM {self.tab_name} WHERE id IN ({id_seq}) AND key IN ({name_seq})"
        args = ids + self.cf_names
        id_cf_values = {}
        id_cf_names = {}
        with connect(self.db_name) as db:
            con = db.connection
            cur = con.cursor()
            cur.execute(sql, args)

            for row in cur.fetchall():
                name = row[0]
                value = row[1]
                dbId = row[2]

                cur_row = id_cf_values.get(dbId, [])
                names = id_cf_names.get(dbId, [])

                cur_row.append(value)
                names.append(name)

                id_cf_values[dbId] = cur_row
                id_cf_names[dbId] = names

        # Make sure that all the rows are ordered in the same way
        key = list(id_cf_names.keys())[0]
        self._cf_names = id_cf_names[key]
        cf_name_col = {n: i for i, n in enumerate(id_cf_names[key])}

        cf_matrix = np.zeros((len(id_cf_values), len(cf_name_col)))
        row = 0
        for row, dbId in enumerate(ids):
            cf_values = id_cf_values[dbId]
            cf_names = id_cf_names[dbId]
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

    def name(self):
        return "E_DFT (eV/atom)"

    def __call__(self, ids: List[int]) -> np.ndarray:
        id_seq = ','.join(['?']*len(ids))
        sql = f"SELECT value, id FROM number_key_values WHERE key='final_struct_id' AND id IN ({id_seq})"

        # Map beetween the initial structure and the index in the id list
        init_struct_idx = {idx: i for i, idx in enumerate(ids)}

        with connect(self.db_name) as db:
            con = db.connection
            cur = con.cursor()
            cur.execute(sql, ids)
            final_struct_ids = []

            # Map the between final struct id and initial struct id
            init_struct_ids = {}
            for row in cur.fetchall():
                final_id, init_id = row
                final_struct_ids.append(final_id)
                init_struct_ids[final_id] = init_id
            final_struct_id_seq = ','.join(['?']*len(final_struct_ids))
            sql = f"SELECT id, energy, natoms FROM systems WHERE id IN ({final_struct_id_seq})"
            cur.execute(sql, final_struct_ids)
            energies = np.zeros(len(final_struct_ids))
            for row in cur.fetchall():
                final_id, energy, natoms = row
                init_id = init_struct_ids[final_id]
                idx = init_struct_idx[init_id]
                energies[idx] = energy/natoms
        return energies
