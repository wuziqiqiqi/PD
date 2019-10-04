from itertools import product, permutations, combinations
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
        conc = self.setting.concentration
        try:
            x = conc.get_random_concentration(nib=nib)
            x_int = conc.conc_in_int(nib, x)
            x_from_int = conc.to_float_conc(nib, x_int)
            if not conc.is_valid_conc(x_from_int):
                return False
        except Exception:
            valid = False
        return valid


class DistanceBetweenFacetsFilter(CellFilter):
    def __init__(self, ratio):
        self.ratio = ratio

    def _distance_between_facets(self, cell, span):
        v1 = cell[span[0], :]
        v2 = cell[span[1], :]
        normal = np.cross(v1, v2)
        normal /= np.sqrt(normal.dot(normal))

        third_vec = set([0, 1, 2]) - set(span)
        third_vec = list(third_vec)[0]
        d = normal.dot(cell[third_vec, :])
        return abs(d)

    def __call__(self, cell):
        dists = []
        for span in combinations([0, 1, 2], 2):
            dists.append(self._distance_between_facets(cell, span))

        d_min = min(dists)
        d_max = max(dists)
        return d_max/d_min < self.ratio


class VolumeToSurfaceRatioFilter(CellFilter):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, cell):
        vol = np.linalg.det(cell)
        surf = 0.0
        for span in combinations([0, 1, 2], r=2):
            v1 = cell[span[0], :]
            v2 = cell[span[1], :]
            normal = np.cross(v1, v2)
            area = np.abs(normal.dot(normal))
            surf += 2*area

        factor = area/(6.0*vol**(2.0/3.0))
        return factor < self.ratio


class AngleFilter(CellFilter):
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, cell):
        cos_max = np.cos(self.max_angle*np.pi/180.0)
        cos_min = np.cos(self.min_angle*np.pi/180.0)

        cos_a = []
        for vec in combinations([0, 1, 2], r=2):
            cos_a.append(cell[vec[0], :].dot(cell[vec[1], :]))

        max_ok = all(x > cos_max for x in cos_a)
        min_ok = all(x < cos_min for x in cos_a)
        return max_ok and min_ok
