import numpy as np
from scipy.spatial import cKDTree as KDTree
from itertools import combinations


class ClusterExtractor(object):
    """
    Class that extracts clusters from an atoms object. Given a
    reference index. This class uses the singular values of the
    inner product matrix of the coordinates of the atoms forming
    a cluster as a unique footprint of the cluster.

    Parameters
    atoms: Atoms
        Atoms object to use
    """
    def __init__(self, atoms):
        self.atoms = atoms
        self.tree = KDTree(self.atoms.get_positions())
        self.svds = []
        self.tol = 1E-6

    def extract(self, ref_indx=0, size=2, cutoff=4.0):
        """
        Extract single clusters

        ref_indx: int
            Reference index

        size: int
            Cluster size

        cutoff: float
            Maximum cutoff
        """
        self.svds = []
        x = self.atoms.get_positions()[ref_indx, :]
        indices = self.tree.query_ball_point(x, cutoff)
        indices.remove(ref_indx)
        return self._group_clusters(ref_indx, indices, size, cutoff)

    def _get_type(self, singular):
        if self.svds:
            diff = [np.sum((x - singular)**2) for x in self.svds]
            min_group = np.argmin(diff)
            if np.sqrt(diff[min_group]) < self.tol:
                return min_group

        self.svds.append(singular)
        return len(self.svds) - 1

    def _group_clusters(self, ref_indx, indices, size, cutoff):
        """
        Group sites in clusters based on their SVD
        """
        pos = self.atoms.get_positions()
        clusters = []
        for comb in combinations(indices, r=size-1):
            all_indices = [ref_indx] + list(comb)

            d = self.get_cluster_diameter(all_indices)
            if d > cutoff:
                continue
            X = pos[all_indices, :]
            X -= np.mean(X, axis=0)
            X = X.dot(X.T)
            s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
            group = self._get_type(s)

            all_indices = self._order_by_internal_distances(all_indices)
            if group == len(clusters):
                clusters.append([all_indices])
            else:
                clusters[group].append(all_indices)
        return clusters

    def _order_by_internal_distances(self, cluster):
        """
        Order the indices by internal distances
        """
        dists = self._get_internal_distances(cluster)
        zipped = sorted(list(zip(dists, cluster)), reverse=True)
        return [x[1] for x in zipped]

    def view_subclusters(self, single_cluster):
        """
        Visualize clusters
        """
        from ase.visualize import view
        images = [self.atoms[x] for x in single_cluster]
        view(images)

    def _get_internal_distances(self, sub_cluster):
        """
        Calculate all internal distances of the cluster
        """
        dists = []
        for indx in sub_cluster:
            d = self.atoms.get_distances(indx, sub_cluster)
            dists.append(sorted(d.tolist(), reverse=True))
        return dists

    def equivalent_sites(self, sub_cluster):
        """
        Finds the equivalent sites of a subcluster
        """
        dists = self._get_internal_distances(sub_cluster)
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
                if any(equiv in m):
                    m.update(equiv)
                    found_group = True
            if not found_group:
                merged.append(set(equiv))
        return [list(x) for x in merged]

    def get_cluster_diameter(self, sub_cluster):
        """
        Return the diameter of the sub cluster
        """
        internal_dists = self._get_internal_distances(sub_cluster)
        max_dists = [x[0] for x in internal_dists]
        return max(max_dists)
