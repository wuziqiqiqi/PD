from itertools import filterfalse, product
import numpy as np


class ClusterGenerator(object):
    """
    Class for generating cluster info that is independent of the
    template
    """
    def __init__(self, prim_cell):
        self.prim = prim_cell
        self.shifts = self.prim.get_positions()

    def eucledian_distance_vec(self, x1, x2):
        """
        Eucledian distance between to vectors

        Parameters:
        x1: array int
            First vector
        x2: array int
            Second vector
        """
        cellT = self.prim.get_cell().T
        euc1 = cellT.dot(x1[:3]) + self.shifts[x1[3]]
        euc2 = cellT.dot(x2[:3]) + self.shifts[x2[3]]
        return euc2 - euc1

    def cartesian(self, x):
        cellT = self.prim.get_cell().T
        return cellT.dot(x[:3]) + self.shifts[x[3]]

    def eucledian_distance(self, x1, x2):
        d = self.eucledian_distance_vec(x1, x2)
        return np.sqrt(np.sum(d**2))

    @property
    def shortest_diag(self):
        shortest = None
        cellT = self.prim.get_cell().T
        for w in product([-1, 0, 1], repeat=3):
            if all(x == 0 for x in w):
                continue

            diag = cellT.dot(w)
            length = np.sqrt(np.sum(diag**2))
            if shortest is None or length < shortest:
                shortest = length
        return shortest

    @property
    def num_sub_lattices(self):
        return len(self.prim)

    def sites_within_cutoff(self, cutoff, ref_lattice=0):
        min_diag = self.shortest_diag
        max_int = int(cutoff/min_diag) + 1
        x0 = [0, 0, 0, ref_lattice]
        sites = filterfalse(
            lambda x: (self.eucledian_distance(x0, x) > cutoff
                       or sum(abs(y) for y in x) == 0),
            product(range(-max_int, max_int+1),
                    range(-max_int, max_int+1),
                    range(-max_int, max_int+1),
                    range(self.num_sub_lattices)))
        return sites

    def generate(self, size, cutoff):
        clusters = []
        fp = []
        