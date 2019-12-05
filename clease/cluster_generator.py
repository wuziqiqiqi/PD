from itertools import filterfalse, product
import numpy as np
from clease.cluster_fingerprint import ClusterFingerprint
from scipy.spatial import cKDTree as KDTree
from ase.geometry import wrap_positions


class ClusterGenerator(object):
    """
    Class for generating cluster info that is independent of the
    template
    """
    def __init__(self, prim_cell):
        self.with_cutoff = SitesWithinCutoff(self)
        self.prim = prim_cell
        self.shifts = np.zeros((len(prim_cell), 3))
        pos = self.prim.get_positions()
        for i, atom in enumerate(prim_cell):
            self.shifts[atom.tag, :] = pos[i, :]
        self.prim_cell_invT = np.linalg.inv(self.prim.get_cell().T)

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
        euc1 = cellT.dot(x1[:3]) + self.shifts[x1[3], :]
        euc2 = cellT.dot(x2[:3]) + self.shifts[x2[3], :]
        return euc2 - euc1

    def cartesian(self, x):
        cellT = self.prim.get_cell().T
        return cellT.dot(x[:3]) + self.shifts[x[3]]

    def get_four_vector(self, pos, lattice):
        """Return the four vector of an atom."""
        if lattice is None:
            lattice = self.get_lattice(pos)

        pos -= self.shifts[lattice]
        int_pos = self.prim_cell_invT.dot(pos)
        four_vec = np.zeros(4, dtype=int)
        four_vec[:3] = np.round(int_pos).astype(int)
        four_vec[3] = lattice
        return four_vec

    def get_lattice(self, pos):
        """
        Return the corresponding sublattice of a cartesian position

        Parameters:
        pos: array of length 3
            Cartesian position
        """
        reshaped = np.reshape(pos, (1, 3))
        wrapped = wrap_positions(reshaped, self.prim.get_cell())
        diff_sq = np.sum((wrapped[0, :] - self.shifts)**2, axis=1)
        return np.argmin(diff_sq)

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

    def sites_within_cutoff(self, cutoff, x0=[0, 0, 0, 0]):
        min_diag = self.shortest_diag
        max_int = int(cutoff/min_diag) + 1

        def filter_func(x):
            d = self.eucledian_distance(x0, x)
            return d > cutoff or d < 1E-5

        sites = filterfalse(
            filter_func,
            product(range(-max_int, max_int+1),
                    range(-max_int, max_int+1),
                    range(-max_int, max_int+1),
                    range(self.num_sub_lattices)))
        return sites

    def get_fp(self, X):
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
        assert len(off_diag) == N*(N-1)/2
        inner = np.array(sorted(diag, reverse=True) + sorted(off_diag))
        fp = ClusterFingerprint(list(inner))
        return fp

    def prepare_within_cutoff(self, cutoff, lattice):
        within_cutoff = {}
        x0 = [0, 0, 0, lattice]
        sites = self.with_cutoff.get(cutoff, lattice)
        sites = set(map(tuple, sites))
        within_cutoff[tuple(x0)] = sites
        for s in sites:
            nearby = [s1 for s1 in sites
                      if self.eucledian_distance(s1, s) <= cutoff and s1 != s]

            within_cutoff[s] = nearby
        return within_cutoff

    def site_iterator(self, within_cutoff, size, ref_lattice):
        """
        Return an iterator of all combinations of sites within a cutoff

        Parameters:

        within_cutoff: dict
            Dictionary returned by `prepare_within_cutoff`

        size: int
            Cluster size

        ref_lattice: int
            Reference lattice
        """
        x0 = (0, 0, 0, ref_lattice)

        def recursive_yield(remaining, current):
            if len(current) == size - 1:
                yield current
            else:
                for next_item in remaining:
                    # Avoid double counting
                    if any(next_item <= v for v in current):
                        continue
                    rem = set.intersection(remaining, within_cutoff[next_item])
                    yield from recursive_yield(rem, current+[next_item])

        for v in within_cutoff[x0]:
            rem = within_cutoff[x0].intersection(within_cutoff[v])
            yield from recursive_yield(rem, [v])

    def generate(self, size, cutoff, ref_lattice=0):
        clusters = []
        all_fps = []
        v0 = [0, 0, 0, ref_lattice]
        x0 = np.array(self.cartesian(v0))
        cutoff_lut = self.prepare_within_cutoff(cutoff, ref_lattice)
        for comb in self.site_iterator(cutoff_lut, size, ref_lattice):
            X = [x0] + [self.cartesian(v) for v in comb]
            fp = self.get_fp(X)
            new_item = [v0] + [list(x) for x in comb]

            # Find the group
            try:
                group = all_fps.index(fp)
                clusters[group].append(new_item)
            except ValueError:
                # Does not exist, create a new group
                clusters.append([new_item])
                all_fps.append(fp)

        # Order the indices by internal indices
        for cluster in clusters:
            for i, f in enumerate(cluster):
                ordered_f = self._order_by_internal_distances(f)
                cluster[i] = ordered_f
        return clusters, all_fps

    def _get_internal_distances(self, figure):
        """
        Return all the internal distances of a figure
        """
        dists = []
        for x0 in figure:
            d = []
            for x1 in figure:
                dist = self.eucledian_distance(x0, x1)
                dist = dist.round(decimals=6)
                d.append(dist.tolist())
            dists.append(sorted(d, reverse=True))
        return dists

    def _order_by_internal_distances(self, figure):
        """Order the indices by internal distances."""
        dists = self._get_internal_distances(figure)
        zipped = sorted(list(zip(dists, figure)), reverse=True)
        return [x[1] for x in zipped]

    def to_atom_index(self, cluster, lut):
        """
        Convert the integer vector representation to an atomic index

        Parameters:
        clusters: list
            List of clusters (in integer vector representation)

        template: Atoms
            Template atoms to use

        lut: dict
            Look up table for 4-vectors to indices
        """
        return [[lut[tuple(ivec)] for ivec in fig] for fig in cluster]

    def equivalent_sites(self, figure):
        """Find the equivalent sites of a figure."""
        dists = self._get_internal_distances(figure)
        equiv_sites = []
        for i in range(len(dists)):
            for j in range(i+1, len(dists)):
                if np.allclose(dists[i], dists[j]):
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

    def get_max_distance(self, figure):
        """Return the maximum distance of a figure."""
        internal_dists = self._get_internal_distances(figure)
        max_dists = [x[0] for x in internal_dists]
        return max(max_dists)


class SitesWithinCutoff(object):
    """
    Class that keeps track of nearby sites. It automatically re-calculates if
    the cutoff changes. If it can re-use a previous result it will do that.
    """
    def __init__(self, generator):
        self.generator = generator
        self.cutoff = 0.0
        self.pre_calc = {}

    def must_generate(self, cutoff, ref_lattice):
        """
        Check if we can re-use a precalculated result
        """
        if abs(cutoff - self.cutoff) > 1e-10:
            return True
        return ref_lattice not in self.pre_calc.keys()

    def get(self, cutoff, ref_lattice):
        """
        Return sites within the cutoff
        """
        if self.must_generate(cutoff, ref_lattice):
            sites = self.generator.sites_within_cutoff(
                cutoff, [0, 0, 0, ref_lattice])
            self.pre_calc[ref_lattice] = list(sites)
            self.cutoff = cutoff
        return self.pre_calc[ref_lattice]
