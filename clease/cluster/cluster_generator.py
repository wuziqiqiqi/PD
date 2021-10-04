from itertools import filterfalse, product
from typing import List, Tuple, Dict, Set, Iterator, Iterable
import numpy as np
from ase import Atoms
from ase.geometry import wrap_positions
from clease.datastructures import FourVector, Figure
from .cluster import Cluster
from .cluster_fingerprint import ClusterFingerprint

__all__ = ('ClusterGenerator', 'SitesWithinCutoff')


class ClusterGenerator:
    """
    Class for generating cluster info that is independent of the
    template
    """

    def __init__(self, prim_cell: Atoms) -> None:
        self.with_cutoff = SitesWithinCutoff(self)
        # Initialize some variables which are updated when setting "prim".
        self.prim_cell_T = None
        self.prim_cell_invT = None

        self.prim = prim_cell

    @property
    def prim(self) -> Atoms:
        return self._prim

    @prim.setter
    def prim(self, other: Atoms) -> None:
        if not isinstance(other, Atoms):
            raise TypeError("prim must be an Atoms object")
        self._prim = other
        # Get the transposed cell as a contiguous array
        self.prim_cell_T = np.array(other.get_cell().T)
        self.prim_cell_invT = np.linalg.inv(self.prim_cell_T)

    def __eq__(self, other: 'ClusterGenerator') -> bool:
        if not isinstance(other, ClusterGenerator):
            return NotImplemented

        return np.allclose(self.prim_cell_invT, other.prim_cell_invT) and self.prim == other.prim

    def eucledian_distance_vec(self, x1: FourVector, x2: FourVector) -> np.ndarray:
        """
        Eucledian distance between to FourVectors in cartesian coordinates.

        :param x1: First vector
        :param x2: Second vector
        """

        euc1 = x1.to_cartesian(self.prim, transposed_cell=self.prim_cell_T)
        euc2 = x2.to_cartesian(self.prim, transposed_cell=self.prim_cell_T)
        return euc2 - euc1

    def to_four_vector(self, cartesian: np.ndarray, sublattice: int = None) -> FourVector:
        """Translate a position in Cartesian coordinates to its FourVector"""
        if not cartesian.ndim == 1:
            # XXX: This could instead return a List[FourVector] if we pass a 2d
            # set of coordinates.
            raise ValueError('Can only translate 1 position at a time.')

        if sublattice is None:
            sublattice = self.get_lattice(cartesian)
        cartesian = cartesian - self.prim.positions[sublattice, :]

        int_pos = np.round(self.prim_cell_invT.dot(cartesian)).astype(int)
        return FourVector(*int_pos, sublattice)

    def to_cartesian(self, *four_vectors: FourVector) -> np.ndarray:
        """Convert one or more FourVectors into their Cartesian coordinates."""
        pos = np.zeros((len(four_vectors), 3))
        for ii, fv in enumerate(four_vectors):
            pos[ii, :] = fv.to_cartesian(self.prim, transposed_cell=self.prim_cell_T)
        return pos

    def get_lattice(self, pos: np.ndarray) -> int:
        """
        Return the corresponding sublattice of a cartesian position

        :param pos: array of length 3 Cartesian position
        """
        shifts = self.prim.get_positions()
        reshaped = np.reshape(pos, (1, 3))
        wrapped = wrap_positions(reshaped, self.prim.get_cell())
        diff_sq = np.sum((wrapped[0, :] - shifts)**2, axis=1)
        return np.argmin(diff_sq)

    def eucledian_distance(self, x1: FourVector, x2: FourVector) -> float:
        d = self.eucledian_distance_vec(x1, x2)
        return np.linalg.norm(d, ord=2)

    @property
    def shortest_diag(self) -> float:
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
    def num_sub_lattices(self) -> int:
        return len(self.prim)

    def sites_within_cutoff(self, cutoff: float, x0: FourVector) -> Iterable[FourVector]:
        min_diag = self.shortest_diag
        max_int = int(cutoff / min_diag) + 1

        def filter_func(fv: FourVector) -> bool:
            d = self.eucledian_distance(x0, fv)
            return d > cutoff or d < 1E-5

        all_sites = product(range(-max_int, max_int + 1), range(-max_int, max_int + 1),
                            range(-max_int, max_int + 1), range(self.num_sub_lattices))

        return filterfalse(filter_func, (FourVector(*site) for site in all_sites))

    @staticmethod
    def get_fp(X: List[np.ndarray]) -> ClusterFingerprint:
        """
        Generate finger print given a position matrix
        """
        X = np.array(X)
        com = np.mean(X, axis=0)
        X -= com
        X = X.dot(X.T)
        diag = np.diagonal(X)
        off_diag = []
        for i in range(1, X.shape[0]):
            off_diag += np.diagonal(X, offset=i).tolist()
        N = X.shape[0]
        assert len(off_diag) == N * (N - 1) / 2
        inner = np.array(sorted(diag, reverse=True) + sorted(off_diag))
        fp = ClusterFingerprint(fp=list(inner))
        return fp

    def prepare_within_cutoff(self, cutoff: float,
                              lattice: int) -> Dict[FourVector, Set[FourVector]]:

        within_cutoff = {}
        x0 = FourVector(0, 0, 0, lattice)
        sites = self.with_cutoff.get(cutoff, lattice)
        sites = set(sites)
        within_cutoff[x0] = sites
        for s in sites:
            nearby = set(s1 for s1 in sites if s1 != s and self.eucledian_distance(s1, s) <= cutoff)

            within_cutoff[s] = nearby
        return within_cutoff

    def generate(self,
                 size: int,
                 cutoff: float,
                 ref_lattice: int = 0) -> Tuple[List[List[Figure]], List[ClusterFingerprint]]:
        clusters = []
        all_fps = []
        v0 = FourVector(0, 0, 0, ref_lattice)

        cutoff_lut = self.prepare_within_cutoff(cutoff, ref_lattice)
        for comb in site_iterator(cutoff_lut, size, ref_lattice):

            new_figure = Figure([v0.copy()] + comb)
            # The entries in the figure must be ordered, due to equiv_sites
            # TODO: This ordereding should not be necessary.
            new_figure = self._order_by_internal_distances(new_figure)

            X = [
                fv.to_cartesian(self.prim, transposed_cell=self.prim_cell_T)
                for fv in new_figure.components
            ]
            fp = self.get_fp(X)

            # Find the group
            try:
                group = all_fps.index(fp)
                clusters[group].append(new_figure)
            except ValueError:
                # Does not exist, create a new group
                clusters.append([new_figure])
                all_fps.append(fp)

        return clusters, all_fps

    def _get_internal_distances(self, figure: Figure) -> List[List[float]]:
        """
        Return all the internal distances of a figure
        """
        dists = []
        for x0 in figure.components:
            tmp_dist = []
            for x1 in figure.components:
                dist = self.eucledian_distance(x0, x1)
                # float() to ensure we're not working on a NumPy floating number
                dist = round(float(dist), 6)
                tmp_dist.append(dist)
            dists.append(sorted(tmp_dist, reverse=True))
        return dists

    def _order_by_internal_distances(self,
                                     figure: Figure) \
            -> Figure:
        """Order the Figure by internal distances. Returns a new instance of the Figure."""
        dists = self._get_internal_distances(figure)
        fvs = figure.components
        zipped = sorted(list(zip(dists, fvs)), reverse=True)
        return Figure((x[1] for x in zipped))

    def to_atom_index(self, cluster: Cluster, lut: Dict[FourVector, int]) -> List[List[int]]:
        """
        Convert the integer vector representation to an atomic index

        :param clusters: List of clusters (in integer vector representation)

        :param template: Template atoms to use

        :param lut: Look up table for 4-vectors to indices
        """
        return [[lut[fv] for fv in fig.components] for fig in cluster.figures]

    def equivalent_sites(self, figure: Figure) -> List[List[int]]:
        """Find the equivalent sites of a figure."""
        dists = self._get_internal_distances(figure)
        equiv_sites = []
        for i, d1 in enumerate(dists):
            for j in range(i + 1, len(dists)):
                d2 = dists[j]
                if np.allclose(d1, d2):
                    equiv_sites.append((i, j))

        # Merge pairs into groups
        merged = []
        for equiv in equiv_sites:
            found_group = False
            for m in merged:
                if any(x in m for x in equiv):
                    m.update(equiv)
                    found_group = True
            if not found_group:
                merged.append(set(equiv))
        return [list(x) for x in merged]

    def get_max_distance(self, figure: Figure):
        """Return the maximum distance of a figure."""
        internal_dists = self._get_internal_distances(figure)
        max_dists = [x[0] for x in internal_dists]
        return max(max_dists)


class SitesWithinCutoff:
    """
    Class that keeps track of nearby sites. It automatically re-calculates if
    the cutoff changes. If it can re-use a previous result it will do that.
    """

    def __init__(self, generator: ClusterGenerator) -> None:
        self.generator = generator
        self.cutoff = 0.0
        self.pre_calc = {}

    def must_generate(self, cutoff: float, ref_lattice: int) -> bool:
        """
        Check if we can re-use a precalculated result
        """
        if abs(cutoff - self.cutoff) > 1e-10:
            return True
        return ref_lattice not in self.pre_calc.keys()

    def get(self, cutoff: float, ref_lattice: int) -> Dict[int, List[Tuple[int]]]:
        """
        Return sites within the cutoff
        """
        if self.must_generate(cutoff, ref_lattice):
            sites = self.generator.sites_within_cutoff(cutoff, FourVector(0, 0, 0, ref_lattice))
            self.pre_calc[ref_lattice] = list(sites)
            self.cutoff = cutoff
        return self.pre_calc[ref_lattice]


def site_iterator(within_cutoff: Dict[FourVector, Set[FourVector]], size: int,
                  ref_lattice: int) -> Iterator[List[FourVector]]:
    """
    Return an iterator of all combinations of sites within a cutoff

    :param within_cutoff: Dictionary returned by `prepare_within_cutoff`

    :param size: Cluster size

    :param ref_lattice: Reference lattice
    """
    x0 = FourVector(0, 0, 0, ref_lattice)

    def recursive_yield(remaining, current):
        if len(current) == size - 1:
            yield current
        else:
            for next_item in remaining:
                # Avoid double counting
                if any(next_item <= v for v in current):
                    continue
                rem = set.intersection(remaining, within_cutoff[next_item])
                yield from recursive_yield(rem, current + [next_item])

    for v in within_cutoff[x0]:
        rem = within_cutoff[x0].intersection(within_cutoff[v])
        yield from recursive_yield(rem, [v])
