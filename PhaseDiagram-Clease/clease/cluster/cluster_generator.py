from itertools import product
from typing import List, Tuple, Dict, Set, Iterator, Iterable, Union, Sequence
from math import sqrt
import numpy as np
from scipy.spatial.distance import cdist
from ase import Atoms
from clease import tools
from clease.datastructures import FourVector, Figure
from .cluster import Cluster
from .cluster_fingerprint import ClusterFingerprint

__all__ = ("ClusterGenerator", "SitesWithinCutoff")


class ClusterGenerator:
    """
    Class for generating cluster info that is independent of the
    template
    """

    def __init__(self, prim_cell: Atoms) -> None:
        self.with_cutoff = SitesWithinCutoff(self)
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
        self.prim_cell_T = np.ascontiguousarray(other.get_cell().T)
        self.prim_cell_invT = np.linalg.inv(self.prim_cell_T)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ClusterGenerator):
            return NotImplemented

        return np.allclose(self.prim_cell_invT, other.prim_cell_invT) and self.prim == other.prim

    def _fv_to_cart(self, fv: FourVector) -> np.ndarray:
        """Helper function to convert a FourVector object to cartesian using the internal
        primitive cell."""
        return fv.to_cartesian(self.prim, transposed_cell=self.prim_cell_T)

    def _as_euc(self, x: Union[np.ndarray, FourVector]) -> np.ndarray:
        """Helper function to translate vector to NumPy array format"""
        if isinstance(x, FourVector):
            return self._fv_to_cart(x)
        # Assume it's already in cartesian coordinates
        return x

    def many_to_four_vector(
        self, cartesian: np.ndarray, sublattices: Sequence[int]
    ) -> List[FourVector]:
        """Translate many positions into FourVector's"""

        if cartesian.ndim != 2:
            raise ValueError(f"Cartesian must be 2 dimensional, got {cartesian.ndim:d}")

        if cartesian.shape[1] != 3:
            raise ValueError(
                "Cartesian should have a length of 3 on the 2nd dimension, "
                f"got {cartesian.shape[1]:d}."
            )

        if len(sublattices) != len(cartesian):
            raise ValueError(
                "Number of sublattices should be same as number of cartesian positions"
            )

        delta = cartesian - self.prim.positions[sublattices, :]
        inv = self.prim_cell_invT.dot(delta.T).T
        # Round off, and cast to integer
        ints = np.rint(inv).astype(int)  # dtype: np.int64

        # One position per sublattice
        assert len(sublattices) == len(ints)
        # Iterating the array as a list is more efficient than iterating the numpy array.
        return [FourVector(*vals, subl) for vals, subl in zip(ints.tolist(), sublattices)]

    def to_four_vector(self, cartesian: np.ndarray, sublattice: int = None) -> FourVector:
        """Translate a position in Cartesian coordinates to its FourVector"""
        if cartesian.ndim != 1:
            raise ValueError(f"Cartesian positions must be 1-dimensional, got {cartesian.ndim:d}")

        if sublattice is None:
            sublattice = self.get_lattice(cartesian)

        return self.many_to_four_vector(cartesian[None, :], [sublattice])[0]

    def to_cartesian(self, *four_vectors: FourVector) -> np.ndarray:
        """Convert one or more FourVectors into their Cartesian coordinates."""
        # We calculate the dot product here, instead of using the fv.to_cartesian, in order
        # to better utilize the NumPy vectorization.
        # So first we allocate the necessary arrays
        X = np.zeros((len(four_vectors), 3), dtype=int)
        for ii, fv in enumerate(four_vectors):
            X[ii, :] = fv.ix, fv.iy, fv.iz
        subl = [fv.sublattice for fv in four_vectors]
        return self.prim_cell_T.dot(X.T).T + self.prim.positions[subl]

    def get_lattice(self, pos: np.ndarray) -> int:
        """
        Return the corresponding sublattice of a cartesian position

        :param pos: array of length 3 Cartesian position
        """
        shifts = self.prim.get_positions()
        reshaped = np.reshape(pos, (1, 3))
        wrapped = tools.wrap_positions_3d(
            reshaped, self.prim.get_cell(), cell_T_inv=self.prim_cell_invT
        )
        diff_sq = np.sum((wrapped[0, :] - shifts) ** 2, axis=1)
        return int(np.argmin(diff_sq))

    def eucledian_distance(
        self,
        x1: Union[np.ndarray, FourVector],
        x2: Union[np.ndarray, FourVector],
    ) -> float:
        """
        Eucledian distance between to FourVectors in cartesian coordinates.
        Either the arrays are given as a FourVector object, in which case they are translated
        into the cartesian coordiates, or they are assumed to already be cartesian coordinates.
        This allows for either passing in a pre-calculated euclidian vector,
        or the FourVector itself.

        :param x1: First vector
        :param x2: Second vector
        """

        # Find the distance vector
        euc1 = self._as_euc(x1)
        euc2 = self._as_euc(x2)
        return _1d_array_norm(euc2 - euc1)

    @property
    def shortest_diag(self) -> float:
        cellT = self.prim_cell_T

        def _iter_diags() -> Iterable[float]:
            """Helper function to iterate all diagonals"""
            for w in product([-1, 0, 1], repeat=3):
                if all(x == 0 for x in w):
                    continue

                diag: np.ndarray = cellT.dot(w)
                length: float = _1d_array_norm(diag)
                yield length

        # Find the shortest length
        shortest = min(_iter_diags())
        return shortest

    @property
    def num_sub_lattices(self) -> int:
        return len(self.prim)

    def sites_within_cutoff(self, cutoff: float, x0: FourVector) -> Iterable[FourVector]:
        min_diag = self.shortest_diag
        max_int = int(cutoff / min_diag) + 1

        # Pre-calculated cartesian representation
        cart0 = self._fv_to_cart(x0)

        def filter_func(fv: FourVector) -> bool:
            d = self.eucledian_distance(cart0, fv)
            return 1e-5 < d < cutoff

        all_sites = product(
            range(-max_int, max_int + 1),
            range(-max_int, max_int + 1),
            range(-max_int, max_int + 1),
            range(self.num_sub_lattices),
        )

        return filter(filter_func, (FourVector(*site) for site in all_sites))

    def get_fp(self, figure: Figure) -> ClusterFingerprint:
        """Generate finger print for a Figure object.

        Args:
            figure (Figure): The figure to find the finger print from.

        Returns:
            ClusterFingerprint: The finger print of the cluster.
        """
        X = figure.to_cartesian(self.prim, transposed_cell=self.prim_cell_T)
        return positions_to_fingerprint(X)

    def prepare_within_cutoff(
        self, cutoff: float, lattice: int
    ) -> Dict[FourVector, Set[FourVector]]:
        """Prepare all sites which are within the cutoff sphere. Note, this only prepares sites
        which are pair-wise within the cutoff, and does not consider the distance to the
        center-of-mass. This needs to be checked for the individual figure which is created from
        these sites."""

        within_cutoff = {}
        x0 = FourVector(0, 0, 0, lattice)
        sites = set(self.with_cutoff.get(cutoff, lattice))
        within_cutoff[x0] = sites

        dist_fnc = self.eucledian_distance  # Cache the distance function
        for s in sites:
            # Precalculate the cartesian representation of the FourVector
            cart_s = self._fv_to_cart(s)
            nearby = set(s1 for s1 in sites if s1 != s and dist_fnc(s1, cart_s) <= cutoff)

            within_cutoff[s] = nearby
        return within_cutoff

    def figure_iterator(self, size: int, cutoff: float, ref_lattice: int) -> Iterator[Figure]:
        """Iterate all possible figures of a given size, cutoff radius, and reference lattice.
        All figures are guaranteed to have a radius smaller than the cutoff radius.

        Args:
            size (int): Size of the clusters to be generated, e.g. 2 for 2-body clusters.
            cutoff (float): Cutoff sphere from the center of mass of the cluster.
            ref_lattice (int, optional): Site which clusters are generated from. Any cluster
            always contains this site.

        Yields:
            Iterator[Figure]: Iterable with all acceptable figures.
        """

        def is_figure_ok(figure: Figure) -> bool:
            """Does the figure obey the cutoff radius?
            The diameter can be larger than the maximum distance between two atoms
            in the cluster, and so the site_iterator can prepare clusters with too large
            diameters.

            Returns:
                bool: The diameter of the cluster is within the cutoff
            """
            return figure.get_diameter(self.prim, transposed_cell=self.prim_cell_T) < cutoff

        cutoff_lut = self.prepare_within_cutoff(cutoff, ref_lattice)
        return filter(is_figure_ok, site_iterator(cutoff_lut, size, ref_lattice))

    def generate(
        self, size: int, cutoff: float, ref_lattice: int
    ) -> Tuple[List[List[Figure]], List[ClusterFingerprint]]:
        """Generate all possible figures of a given size, are within a given cutoff radius
        (from the center of mass of the figure), and from a reference lattice.

        Args:
            size (int): Size of the clusters to be generated, e.g. 2 for 2-body clusters.
            cutoff (float): Cutoff sphere from the center of mass of the cluster.
            ref_lattice (int, optional): Site which clusters are generated from. Any cluster
            always contains this site.

        Returns:
            List[List[Figure]], List[ClusterFingerprint]: The collection of figures and
                their corresponding fingerprints.
        """
        clusters: List[List[Figure]] = []
        all_fps: List[ClusterFingerprint] = []

        for new_figure in self.figure_iterator(size, cutoff, ref_lattice):
            # The entries in the figure must be ordered, due to equiv_sites
            # TODO: This ordereding should not be necessary.
            new_figure = self._order_by_internal_distances(new_figure)

            fp = self.get_fp(new_figure)

            # Find the group
            try:
                group = all_fps.index(fp)
                clusters[group].append(new_figure)
            except ValueError:
                # Does not exist, create a new group
                clusters.append([new_figure])
                all_fps.append(fp)

        return clusters, all_fps

    def _get_internal_distances(self, figure: Figure, sort: bool = False) -> np.ndarray:
        """
        Return all the internal distances of a figure.

        If ``sort`` is True, then distances are sorted from
        largest->smallest distance for each component.

        Returns a 2D array with the internal distances. If the array is *unsorted*,
            then entry ``(i, j)`` is the eucledian distance between figures ``i`` and ``j``.
        """
        comp = figure.components

        # Convert each FourVector component to cartesian coordinates
        x = np.array([self._fv_to_cart(c) for c in comp], dtype=float)
        # Build the distance array, such that elements (i, j) is the
        # Euclidean distance betweens elements i and j.
        arr = cdist(x, x, metric="euclid")
        # Round to 6 digits for numerical stability.
        arr = np.round(arr, 6)
        if sort:
            # fliplr for reversing the sort. Sorted such that each row has largest->smallest
            # distance (i.e. like reverse=True does in Python's sort() method).
            arr = np.fliplr(np.sort(arr, axis=1))
        return arr

    def _order_by_internal_distances(self, figure: Figure) -> Figure:
        """Order the Figure by internal distances. Returns a new instance of the Figure."""
        dists = self._get_internal_distances(figure, sort=True)
        fvs = figure.components

        # Sort according to the figure with the largest internal distances, lexographically.
        # Largest internal distances first.
        order = np.lexsort(dists.T)[::-1]
        return Figure((fvs[i] for i in order))

    def to_atom_index(self, cluster: Cluster, lut: Dict[FourVector, int]) -> List[List[int]]:
        """
        Convert the integer vector representation to an atomic index

        :param clusters: List of clusters (in integer vector representation)

        :param template: Template atoms to use

        :param lut: Look up table for 4-vectors to indices
        """
        # pylint: disable=no-self-use
        return [[lut[fv] for fv in fig.components] for fig in cluster.figures]

    def equivalent_sites(self, figure: Figure) -> List[List[int]]:
        """Find the equivalent sites of a figure."""
        dists = self._get_internal_distances(figure, sort=True)
        equiv_sites = []
        for i, d1 in enumerate(dists):
            for j in range(i + 1, len(dists)):
                d2 = dists[j]
                if np.allclose(d1, d2):
                    equiv_sites.append((i, j))

        # Merge pairs into groups
        merged: List[Set[int]] = []
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
        internal_dists = self._get_internal_distances(figure, sort=False)
        return internal_dists.max()


class SitesWithinCutoff:
    """
    Class that keeps track of nearby sites. It automatically re-calculates if
    the cutoff changes. If it can re-use a previous result it will do that.
    """

    def __init__(self, generator: ClusterGenerator) -> None:
        self.generator = generator
        self.cutoff = 0.0
        # Dictionary mapping a ref lattice to a list of four-vectors
        self.pre_calc: Dict[int, List[FourVector]] = {}

    def must_generate(self, cutoff: float, ref_lattice: int) -> bool:
        """
        Check if we can re-use a precalculated result
        """
        if abs(cutoff - self.cutoff) > 1e-10:
            return True
        return ref_lattice not in self.pre_calc

    def get(self, cutoff: float, ref_lattice: int) -> List[FourVector]:
        """
        Return sites within the cutoff
        """
        if self.must_generate(cutoff, ref_lattice):
            sites = self.generator.sites_within_cutoff(cutoff, FourVector(0, 0, 0, ref_lattice))
            self.pre_calc[ref_lattice] = list(sites)
            self.cutoff = cutoff
        return self.pre_calc[ref_lattice]


def site_iterator(
    within_cutoff: Dict[FourVector, Set[FourVector]], size: int, ref_lattice: int
) -> Iterator[Figure]:
    """
    Return an iterator of all combinations of sites within a cutoff

    :param within_cutoff: Dictionary returned by `prepare_within_cutoff`

    :param size: Cluster size

    :param ref_lattice: Reference lattice
    """
    x0 = FourVector(0, 0, 0, ref_lattice)

    def recursive_yield(remaining, current):
        if len(current) == size - 1:
            # This is the full Figure object
            components = [x0] + current
            fig = Figure(components)
            yield fig
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


def positions_to_fingerprint(X: np.ndarray) -> ClusterFingerprint:
    """
    Generate finger print given a position matrix
    """
    # Center around the center of mass
    X = np.array(X)
    com = np.mean(X, axis=0)
    X -= com
    X = X.dot(X.T)
    # The positions squared
    diag_positions = np.diagonal(X)

    N = X.shape[0]
    # Put all the upper triangle off-diagonals into a 1D array one after another,
    # [off_diag1, off_diag2, ...]
    off_diag_len = N * (N - 1) // 2
    off_diag = np.zeros(off_diag_len)
    offset = 0
    for i in range(1, N):
        diag = np.diagonal(X, offset=i)
        off_diag[offset : (len(diag) + offset)] = diag
        offset += len(diag)
    # Sort positions, such that the largest position is first
    diag_positions = np.sort(diag_positions)[::-1]
    off_diag = np.sort(off_diag)
    # Append the off-diagonal elements after the diagonal, into a 1D array
    inner = np.append(diag_positions, off_diag)
    return ClusterFingerprint(fp=inner)


def _1d_array_norm(arr: np.ndarray) -> float:
    """Calculate the norm of a 1-d vector.
    Assumes the vector is 1-dimensional and real, and will not be checked.

    More efficient norm than ``np.linalg.norm`` for small vectors,
    since we already know the dimensionality, all numbers are real,
    and that the result is scalar.
    """
    return sqrt(arr.dot(arr))
