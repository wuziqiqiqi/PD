from itertools import filterfalse, product
from itertools import combinations
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

    def generate(self, size, cutoff, ref_lattice=0):
        clusters = []
        all_fps = []
        sites = self.sites_within_cutoff(cutoff, ref_lattice=ref_lattice)
        v0 = [0, 0, 0, ref_lattice]
        x0 = np.array(self.cartesian(v0))
        for comb in combinations(sites, r=size-1):
            X = [x0] + [self.cartesian(v) for v in comb]
            fp = self.get_fp(X)
            if np.max(np.sqrt(fp.fp[:size])) > cutoff/2.0:
                continue

            # Find the group
            new_item = [v0] + [list(x) for x in comb]
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

    def to_atom_index(self, clusters, template):
        """
        Convert the integer vector representation to an atomic index

        Parameters:
        clusters: list
            List of clusters (in integer vector representation)

        template: Atoms
            Template atoms to use
        """
        cell = template.get_cell()
        pos = template.get_positions()
        tree = KDTree(pos)
        int_clusters = []

        for cluster in clusters:
            int_cluster = []
            for fig in cluster:
                int_fig = []
                for ivec in fig:
                    x = self.cartesian(ivec)
                    x_mat = np.zeros((1, 3))
                    x_mat[0, :] = x
                    x = wrap_positions(x_mat, cell)
                    d, i = tree.query(x)
                    int_fig.append(i[0])
                int_cluster.append(int_fig)
            int_clusters.append(int_cluster)
        return int_clusters

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
