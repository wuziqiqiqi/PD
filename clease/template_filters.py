from itertools import product, permutations
import numpy as np


class AtomsFilter(object):
    """
    Base class for all template filters that requires a full Atoms object in
    order to decide the validity of the template.
    """

    def __call__(self, atoms):
        """
        The call method has to be implemented in derived classes.
        It accepts an atoms object or a cell object and returns `True` if the
        templates is valid and False if the templates is not valid.
        """
        raise NotImplementedError("Has to be implemented in derived classes.")


class CellFilter(object):
    """
    Base class for all template filters that only require knowledge of the
    cell vectors in order to decide if the template is valid or not.
    """

    def __call__(self, cell):
        raise NotImplementedError("Has to be implemented in derived classes.")


class SkewnessFilter(CellFilter):
    """
    A filter that rejects cells that are too skewed.

    Parameters:

    ratio: float
        Maximum accepted ratio between the longest and the shortest diagonal
    """

    def __init__(self, ratio):
        self.ratio = ratio

    def _get_max_min_diag_ratio(self, cell):
        """Return the ratio between the maximum and the minimum diagonal."""
        diag_lengths = []
        for w in product([-1, 0, 1], repeat=3):
            if np.allclose(w, 0):
                continue
            diag = np.array(w).dot(cell)
            length = np.sqrt(diag.dot(diag))
            diag_lengths.append(length)
        max_length = np.max(diag_lengths)
        min_length = np.min(diag_lengths)
        return max_length/min_length

    def __call__(self, cell):
        diag_ratio = self._get_max_min_diag_ratio(cell)
        return diag_ratio < self.ratio


class EquivalentCellsFilter(CellFilter):
    """
    A filter that rejects templates that can be obtained by a unitary
    transformation of an existing cell.
    """

    def __init__(self, cell_list):
        self.cell_list = cell_list

    def _is_unitary(self, matrix):
        return np.allclose(matrix.T.dot(matrix), np.identity(matrix.shape[0]))

    def _are_equivalent(self, cell1, cell2):
        """Compare two cells to check if they are equivalent.

        It is assumed that the cell vectors are columns of each matrix.
        """
        inv_cell1 = np.linalg.inv(cell1)
        for perm in permutations(range(3)):
            permute_cell = cell2[:, perm]
            R = permute_cell.dot(inv_cell1)
            if self._is_unitary(R):
                return True
        return False

    def __call__(self, cell):
        for existing_cell in self.cell_list:
            if self._are_equivalent(existing_cell, cell):
                return False
        return True


class ValidConcentrationFilter(AtomsFilter):
    """
    A filter that rejects template that has no valid concentration.

    Parameters:

    setting: ClusterExpansionSetting
        Instance of `ClusterExpansionSetting`
    """

    def __init__(self, setting):
        self.setting = setting

    def __call__(self, atoms):
        num_in_template = len(self.setting.atoms)
        num_in_atoms = len(atoms)
        ratio = num_in_atoms/num_in_template
        nib = [len(x)*ratio for x in self.setting.index_by_basis]

        if not np.allclose(nib, np.round(nib)):
            return False
        valid = True
        try:
            x = self.setting.conc.get_random_concentration(nib=nib)
            x_int = self.setting.conc.conc_in_int(nib, x)
            x_from_int = self.setting.conc.to_float_conc(nib, x_int)
            if not np.allclose(x, x_from_int):
                return False
        except Exception:
            valid = False
        return valid
