from clease import _logger
from ase.db import connect
import numpy as np
from clease.tools import add_file_extension


class DataManager(object):
    """
    DataManager is a class for extracting data from CLEASE databases to be
    used to fit ECIs

    Parameters:

    db_name: str
        Name of the database
    """
    def __init__(self, db_name):
        self.db_name = db_name
        self._X = None
        self._y = None
        self._feat_names = None
        self._target_name = None

    def get_data(self, select_cond, feature_getter, target_getter):
        """
        Return the design matrix X and the target data y

        Parameters:

        select_cond: list
            List of tuples with selection conditions
            (e.g. [('converged', '='' True)])

        feature_getter: Callable object
            Callable object that returns a row in the design matrix
            corresponding to an AtomsRow object from the ASE database. It
            also needs a method get_feature_names that returns a name of
            each feature.

        target_getter: Callable object
            Callable object that extracts the target value. The __call__
            method takes the AtomsRow used to construct the row in the design
            matrix

        Example:

        >>> from clease.data_manager import (
        ... CorrelationFunctionGetter, FinalStructEnergyGetter,
        ... DataManager)
        >>> feat_getter = CorrelationFunctionGetter(['c0', 'c1_0',
        ... 'c2_d0000_0_00'], 'polynomial_cf')
        >>> targ_getter = FinalStructEnergyGetter()
        >>> manager = DataManager('somedb.db')
        >>> X, y = manager.get_data([], feat_getter, targ_getter)
        """
        cfm = []
        target = []
        db = connect(self.db_name)
        for row in db.select(select_cond):
            cfm.append(feature_getter(row, db))
            target.append(target_getter(row, db))

        self._X = np.array(cfm, dtype=float)
        self._y = np.array(target)
        self._feat_names = feature_getter.names()
        self._target_name = target_getter.name()
        return self._X, self._y

    def to_csv(self, fname):
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
        header = ','.join(self._feat_names) + ',{}'.format(self._target_name)
        data = np.hstack((self._X,
                          np.reshape(self._y, (len(self._y), -1))))
        np.savetxt(fname, data, delimiter=",", header=header)
        _logger("Dataset exported to {}".format(fname))

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
    """
    def __init__(self, db_name, cf_names, tab_name):
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
            CorrelationFunctionGetter(self.cf_names, self.tab_name),
            FinalStructEnergyGetter())


class CorrelationFunctionGetter(object):
    """
    CorrelationFunctionGetter is a class that extracts
    the correlation functions from an AtomsRow object

    Parameters:

    cf_names: list
        List with the names of the correlation functions

    tab_name: str
        Name of the external table where the correlation functions
        are stored
    """
    def __init__(self, cf_names, tab_name):
        self.cf_names = cf_names
        self.tab_name = tab_name

    def names(self):
        """
        Return a name of each column
        """
        return self.cf_names

    def __call__(self, row, db):
        return [row[self.tab_name][x] for x in self.cf_names]


class FinalStructEnergyGetter(object):
    """
    FinalStructEnergyGetter is a callable class that returns the final energy
    (typically after structure relaxation) corresponding to the passed
    AtomsRow object.
    """
    def name(self):
        return "E_DFT (eV/atom)"

    def __call__(self, row, db):
        final_struct_id = row.get("final_struct_id", None)

        if final_struct_id is None:
            raise ValueError("The passed atoms row {} has no final_struct_id"
                             "".format(row))

        return db.get(final_struct_id).energy/row.natoms
