from clease import _logger
from ase.db import connect
import numpy as np
from clease.tools import add_file_extension
from ase.db.core import parse_selection
from typing import Tuple, List, Dict, Set, Optional, Callable
import sqlite3


class InconsistentDataError(Exception):
    pass


class DataManager(object):
    """
    DataManager is a class for extracting data from CLEASE databases to be
    used to fit ECIs

    :param db_name: Name of the database
    """
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._X = None
        self._y = None
        self._feat_names = None
        self._target_name = None

    def get_ids(self, select_cond: List[tuple]) -> List[int]:
        """
        Return ids in the database that corresponds to the passed
        selection condition.

        :param select_cond: ASE select condition. The database IDs matching
            the select condition is returned.
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
        return ids

    def get_data(self, select_cond: List[tuple],
                 feature_getter: Callable[[List[int]], np.ndarray],
                 target_getter: Callable[[List[int]], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the design matrix X and the target data y

        :param select_cond: List of tuples with selection conditions
            (e.g. [('converged', '='' True)])

        :param feature_getter:
            Callable object that returns design matrix. It
            also needs a method get_feature_names that returns a name of
            each feature. The passed instance should take a list of IDs
            as input argument and return a numpy array corresponding to
            the design matrix

        :param target_getter:
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
        >>> db.write(Atoms('Cu'), external_tables={
        ... 'polynomial_cf': {'c0': 1.0}}, final_struct_id=2, converged=True)
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
        >>> X, y = manager.get_data([('converged', '=', 1)], feat_getter,
        ... targ_getter)
        """
        ids = self.get_ids(select_cond)

        # Extract design matrix and the target values
        cfm = feature_getter(ids)
        target = target_getter(ids)

        self._X = np.array(cfm, dtype=float)
        self._y = np.array(target)
        self._feat_names = feature_getter.names
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

        :param fname: Filename to write to. Typically this should end with .csv
        """
        if self._X is None:
            return
        fname = add_file_extension(fname, 'csv')
        header = f",".join(self._feat_names) + f",{self._target_name}"
        data = np.hstack((self._X,
                          np.reshape(self._y, (len(self._y), -1))))
        np.savetxt(fname, data, delimiter=",", header=header)
        _logger(f"Dataset exported to {fname}")

    def get_matching_names(self, pattern: str) -> List[str]:
        """
        Get names that matches pattern

        :param pattern: Pattern which the string should contain.

        Example:

        If the names are ['abc', 'def', 'gbcr'] and the passed pattern
        is 'bc', then ['abc', 'gbcr'] will be returned
        """
        return [n for n in self._feat_names if pattern in n]

    def get_cols(self, names: List[str]) -> np.ndarray:
        """
        Get all columns corresponding to the names

        :pram names: List of names (e.g. ['c0', 'c1_1'])
        """
        name_idx = {n: i for i, n in enumerate(self._feat_names)}
        indices = [name_idx[n] for n in names]
        return self._X[:, indices]


class CorrFuncEnergyDataManager(DataManager):
    """
    CorrFuncFinalEnergyDataManager is a convenience class provided
    to handle the standard case where the features are correlation functions
    and the target is the DFT energy per atom

    :param db_name: Name of the database being passed

    :param cf_names: List with the correlation function names to extract

    :param tab_name: Name of the table where the correlation functions are
        stored
    """
    def __init__(self, db_name: str, cf_names: List[str],
                 tab_name: str) -> None:
        DataManager.__init__(self, db_name)
        self.tab_name = tab_name
        self.cf_names = cf_names

    def get_data(self, select_cond: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return X and y, where X is the design matrix containing correlation
        functions and y is the DFT energy per atom.

        :param select_cond: List with select conditions for the database
            (e.g. [('converged', '=', True)])
        """
        return DataManager.get_data(
            self, select_cond,
            CorrelationFunctionGetter(self.db_name, self.cf_names,
                                      self.tab_name),
            FinalStructEnergyGetter(self.db_name))


class CorrelationFunctionGetter(object):
    """
    CorrelationFunctionGetter is a class that extracts
    the correlation functions from an AtomsRow object

    :param db_name: Name of the database

    :param cf_names: List with the names of the correlation functions

    :param tab_name: Name of the external table where the correlation
        functions are stored
    """
    def __init__(self, db_name: str, cf_names: List[str],
                 tab_name: str) -> None:
        self.db_name = db_name
        self.cf_names = cf_names
        self.tab_name = tab_name

    @property
    def names(self):
        """
        Return a name of each column
        """
        return self.cf_names

    def __call__(self, ids: List[int]) -> np.ndarray:
        """
        Extracts the design matrix associated with the database IDs. The first
        row in the matrix corresponds to the first item in ids, the second row
        corresponds to the second item in ids etc.

        :param ids: Database IDs of initial structures
        """
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
                if db_id in id_set and name in self.cf_names:
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

    :param db_name: Name of the database
    """
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name

    @property
    def name(self):
        return "E_DFT (eV/atom)"

    def __call__(self, ids: List[int]) -> np.ndarray:
        """
        Extract the final energy of the ids passed. In the returned array, the
        first energy corresponds to the first item in ids, the second energy
        corresponds to the second item in ids etc.

        :param ids: Database ids of initial structures
        """
        sql = "SELECT value, id FROM number_key_values "
        sql += "WHERE key='final_struct_id'"
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

    :param db_name: Name of the database
    """
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name

    @property
    def name(self):
        return "Volume (A^3)"

    def __call__(self, ids: List[int]) -> np.ndarray:
        """
        Extracts the final volume of the ids passed. In the returned array,
        the first volume corresponds to the first item in ids, the second
        volume corresponds to the second item in ids etc.

        :param ids: Database IDs of initial structures
        """
        id_set = set(ids)
        query = "SELECT value, id FROM number_key_values "
        query += "WHERE key='final_struct_id'"

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

    :param db_name: Name of the database being passed

    :param cf_names: List with the correlation function names to extract

    :param tab_name: Name of the table where the correlation functions are
        stored
    """
    def __init__(self, db_name: str, cf_names: List[str],
                 tab_name: str) -> None:
        DataManager.__init__(self, db_name)
        self.tab_name = tab_name
        self.cf_names = cf_names

    def get_data(self,
                 select_cond: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return X and y, where X is the design matrix containing correlation
        functions and y is the volume per atom.

        Parameters:

        select_cond: list
            List with select conditions for the database
            (e.g. [('converged', '=', True)])
        """
        return DataManager.get_data(
            self, select_cond,
            CorrelationFunctionGetter(
                self.db_name, self.cf_names, self.tab_name),
            FinalVolumeGetter(self.db_name))


def extract_num_atoms(cur: sqlite3.Cursor,
                      ids: Set[int]) -> Dict[int, int]:
    """
    Extract the number of atoms for all ids

    :param cur: SQL-cursor object
    :param ids: Set with IDs in the database.
    """
    sql = "SELECT id, natoms FROM systems"
    cur.execute(sql)
    num_atoms = {}
    for db_id, natoms in cur.fetchall():
        if db_id in ids:
            num_atoms[db_id] = natoms
    return num_atoms


class CorrelationFunctionGetterVolDepECI(DataManager):
    """
    Extracts correlation functions, multiplied with a power of the volume per
    atom. The feature names are named according to the correlation function
    names in the database, but a suffix of _Vd is appended. d is an integer
    inticading the power. Thus, if the name is for example c2_d0000_0_00_V2,
    it means that the column contains the correlation function c2_d0000_0_00,
    multiplied by V^2, where V is the volume per atom.

    :param db_name: Name of the database

    :param tab_name: Name of the table where correlation functions are stored

    :param cf_names: Name of the correlation functions that should be extracted

    :param order: Each ECI will be a polynomial in the volume of the passed
        order (default: 0)

    :param properties: List of properties that should be used in fitting. Can
        be energy, pressure, bulk_mod. (default: ['energy', 'pressure']).
        The pressure is always assumed to be zero (e.g. the energies passed
        are for relaxed structures.). All entries in the database are expected
        to have an energy. The remaining properties (e.g. bulk_mod) is not
        required for all structures. In class will pick up and the material
        property for the structures where it is present.
    """
    def __init__(self, db_name: str, tab_name: str, cf_names: List[str],
                 order: Optional[int] = 0,
                 properties: Tuple[str] = ('energy', 'pressure')) -> None:
        self.db_name = db_name
        self.tab_name = tab_name
        self.order = order
        self.cf_names = cf_names
        self.properties = properties
        self._X = None
        self._y = None

    def build(self, ids: List[int]) -> np.ndarray:
        """
        Construct the design matrix and the target value required to fit a
        cluster expansion model to all material properties in self.properties.

        :param ids: List of ids to take into account
        """
        cf_getter = CorrelationFunctionGetter(self.db_name, self.cf_names,
                                              self.tab_name)
        cf = cf_getter(ids)

        volume_getter = FinalVolumeGetter(self.db_name)
        volumes = volume_getter(ids)

        energy_getter = FinalStructEnergyGetter(self.db_name)
        energies = energy_getter(ids)

        target_values = [energies]
        target_val_names = ['energy']

        cf_vol_dep = np.zeros((cf.shape[0], (self.order+1)*cf.shape[1]))
        counter = 0
        self._feat_names = []

        # Construct the part of the design matrix that corresponds to
        # energy fitting (e.g. if there are N energy calculations we construct
        # the firxt N rows of the design matrix)
        for col in range(cf.shape[1]):
            for power in range(self.order+1):
                cf_vol_dep[:, counter] = cf[:, col]*volumes**power
                counter += 1
                self._feat_names.append(cf_getter.names[col] + f"_V{power}")

        # Add pressure data. P = dE/dV = 0 (we assume relaxed structures)
        if 'pressure' in self.properties:
            pressure_cf = np.zeros_like(cf_vol_dep)
            counter = 0

            # Construct the part of the design matrix that corresponds to
            # pressure fitting (e.g. if there are N energy calculations,
            # we construct row N:2*N). Note that the pressure is assumed
            # to be zero
            for col in range(0, cf.shape[1]):
                for p in range(self.order+1):
                    pressure_cf[:, counter] = p*cf[:, col]*volumes**(p-1)
                    counter += 1

            # Update the full design matrix and the target values
            cf_vol_dep = np.vstack((cf_vol_dep, pressure_cf))
            target_values.append(np.zeros(pressure_cf.shape[0]))
            target_val_names.append('pressure')

        id_row = {db_id: i for i, db_id in enumerate(ids)}

        # Add bulk modulus data. B = V*d^2E/dV^2
        if 'bulk_mod' in self.properties:
            bulk_mod = self._extract_key(set(ids), 'bulk_mod')
            bulk_mod_rows = [id_row[k] for k in bulk_mod.keys()]
            bulk_mod_cf = np.zeros((len(bulk_mod), cf_vol_dep.shape[1]))

            counter = 0
            vols = volumes[bulk_mod_rows]

            # Extract the part of the design matrix that corresponds to fitting
            # bulk moduli. If there are Nb calculations that has a bulk modulus
            # we construct row 2*N:2*N+Nb
            for col in range(0, cf.shape[1]):
                for power in range(self.order+1):
                    bulk_mod_cf[:, counter] = vols*cf[bulk_mod_rows, col]
                    bulk_mod_cf[:, counter] *= power*(power-1)*vols**(power-2)
                    counter += 1

            # Update the full design matrix as well as the target values
            cf_vol_dep = np.vstack((cf_vol_dep, bulk_mod_cf))
            target_values.append(bulk_mod.values())
            target_val_names.append('bulk_mod')

        # Assign the global design matrix to the attribute _X, the vector with
        # all target values to _y and construct the name of the target_value by
        # joining all subnames
        self._X = cf_vol_dep
        self._y = np.array([x for values in target_values for x in values])
        self._target_name = '-'.join(target_val_names)

    def _extract_key(self, ids: Set[int], key: str) -> Dict[int, float]:
        """
        Extract a key from the database for the ids in the passed set that
        has a bulk modlus entry. The function returns a dictionary where the
        key is the ID in the database and the value is the extracted quantity

        :param ids: Set of all ids that should be considered

        :param key: Name of the key to be extracted
        """
        query = f"SELECT value, id FROM number_key_values WHERE key='{key}'"

        id_key = {}
        with connect(self.db_name) as db:
            cur = db.connection.cursor()
            cur.execute(query)
            for value, db_id in cur.fetchall():

                # Extract only the ones that are present in the set of ids
                if db_id in ids:
                    id_key[db_id] = value
        return id_key

    def get_data(self,
                 select_cond: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the design matrix and the target values for the entries
        corresponding to select_cond.

        :param select_cond: ASE select condition. The design matrix and the
            target vector will be extracted for rows matching the passed
            condition.
        """
        ids = self.get_ids(select_cond)

        self.build(ids)
        return self._X, self._y
