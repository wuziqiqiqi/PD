import numpy as np
from scipy.spatial import cKDTree as KDTree
from itertools import combinations
from clease.cluster_fingerprint import ClusterFingerprint


class ClusterExtractor(object):
    """
    Extract clusters that contain the suppied atomic index.


    Parameters:

    atoms: Atoms object
        ASE Atoms object
    """

    def __init__(self, atoms):
        self.atoms = atoms
        self.tree = KDTree(self.atoms.get_positions())
        self.inner_prod = []
        self.tol = 1E-6

    def extract(self, ref_indx=0, size=2, cutoff=4.0, ignored_indices=[]):
        """
        Extract all clusters with a given size if they are smaller than the
        cutoff distance.

        This method returns a list of sites (atomic indices) belonging to
        every cluster. If there are *N* clusters, it returns a list of a form
            [cluster_1, cluster_2, cluster_3, cluster_N]
        where cluster_x is a nested list in a form
            cluster1 = [[245, 432, 126], [567, 432, 127], ...]


        Parameters:

        ref_indx: int
            Index to be included in all of the clusters

        size: int
            Cluster size (i.e., number of atoms in a cluster)

        cutoff: float
            Maximum cutoff distance

        ignored_indices: list
            all of the background indices to be ignored when creating clusters
        """
        self.inner_prod = []
        x = self.atoms.get_positions()[ref_indx, :]
        indices = self.tree.query_ball_point(x, cutoff)
        indices.remove(ref_indx)
        indices = list(set(indices) - set(ignored_indices))
        return self._group_clusters(ref_indx, indices, size, cutoff)

    def _get_type(self, fingerprint):
        """Determine cluster type based on flattened inner product matrix."""
        if self.inner_prod:
            try:
                group = self.inner_prod.index(fingerprint)
                return group
            except ValueError:
                pass

        self.inner_prod.append(fingerprint)
        return len(self.inner_prod) - 1

    def _group_clusters(self, ref_indx, indices, size, cutoff):
        """Group sites in clusters based on the inner product matrix."""
        pos = self.atoms.get_positions()
        clusters = []
        for comb in combinations(indices, r=size-1):
            all_indices = [ref_indx] + list(comb)

            d = self.get_max_distance(all_indices)
            if d > cutoff:
                continue
            X = pos[all_indices, :]
            X -= np.mean(X, axis=0)
            X = X.dot(X.T)

            diag = np.diagonal(X)
            off_diag = []
            for i in range(1, X.shape[0]):
                off_diag += np.diagonal(X, offset=i).tolist()
            N = X.shape[0]
            assert len(off_diag) == N*(N-1)/2

            inner = np.array(sorted(diag, reverse=True) + sorted(off_diag))
            fp = ClusterFingerprint(list(inner))
            group = self._get_type(fp)

            all_indices = self._order_by_internal_distances(all_indices)
            if group == len(clusters):
                clusters.append([all_indices])
            else:
                clusters[group].append(all_indices)
        return clusters

    def _order_by_internal_distances(self, clusters):
        """Order the indices by internal distances."""
        dists = self._get_internal_distances(clusters)
        zipped = sorted(list(zip(dists, clusters)), reverse=True)
        return [x[1] for x in zipped]

    def view_figures(self, cluster):
        """Visualize figures of a cluster."""
        from ase.visualize import view
        figures = [self.atoms[x] for x in cluster]
        view(figures)

    def _get_internal_distances(self, figure):
        """Calculate all internal distances of a figure."""
        dists = []
        for indx in figure:
            d = self.atoms.get_distances(indx, figure)
            d = d.round(decimals=6)
            dists.append(sorted(d.tolist(), reverse=True))
        return dists

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
