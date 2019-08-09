class AtomsFilter(object):
    """
    Base class for all template filters that requires a full
    atoms object in order to decide if the template is valid
    or not
    """
    def __call__(self, atoms):
        """
        The call method has to be implemented in derived classes.
        It accepts an atoms object or a cell object and returns True
        if the templates is valid and False if the templates is not valid.
        """
        raise NotImplementedError("Has to be implemented in derived classes!")


class CellFilter(object):
    """
    Base class for all template filters that only require
    knowledge of the cell vectors in order to decide if the
    template is valid or not.
    """
    def __call__(self, cell):
        raise NotImplementedError("Has to be implemented in derived classes")


class SkewnessFilter(CellFilter):
    """
    Rejects cells that are too skew

    Parameters:

    ratio: float
        Maximum accepted ratio between the longest and the
        shortest diagonal
    """
    def __init__(self, ratio):
        self.ratio = ratio

    def _get_max_min_diag_ratio(self, cell):
        """Return the ratio between the maximum and the minimum diagonal."""
        diag_lengths = []
        cell = atoms.get_cell()
        for w in product([-1, 0, 1], repeat=3):
            if np.allclose(w, 0):
                continue
            diag = w.dot(cell)
            length = np.sqrt(diag.dot(diag))
            diag_lengths.append(length)
        max_length = np.max(diag_lengths)
        min_length = np.min(diag_lengths)
        return max_length/min_length

    def __call__(self, cell):
        diag_ratio = self._get_max_min_diag_ratio(cell)
        return diag_ratio < self.ratio 