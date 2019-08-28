from clease.tools import dec_string
from itertools import product
from clease.tools import flatten

class ClusterList(object):
    def __init__(self):
        self.clusters = []

    def append(self, cluster):
        self.clusters.append(cluster)

    def clear(self):
        """Clear the content."""
        self.clusters = []

    @property
    def names(self):
        return list(set([c.name for c in self.clusters]))

    def get_by_name(self, name):
        return [c for c in self.clusters if c.name == name]

    def get_by_size(self, size):
        # Return all clusters with a given size
        return [c for c in self.clusters if c.size == size]

    def get_by_group(self, group):
        """Return all clusters in a given symmetry group."""
        return [c for c in self.clusters if c.group == group]

    def get_equivalent_clusters(self, cluster):
        """Return equivalent clusters in other symmetry groups."""
        equiv = []
        for c in self.clusters:
            if c == cluster:
                equiv.append(c)
        return equiv

    @staticmethod
    def get_cf_names(cluster, num_bf):
        """Return all possible correlation function names.

        Parameters:

        cluster: Cluster
            Instance of cluster class

        num_bf: int
            Number of basis functions
        """
        name = cluster.name
        eq_sites = cluster.equiv_sites
        bf_list = list(range(num_bf))
        cf_names = []
        for dec in product(bf_list, repeat=cluster.size):
            dec_str = dec_string(dec, eq_sites)
            cf_names.append(name + '_' + dec_str)
        return sorted(list(set(cf_names)))

    def get_all_cf_names(self, num_bf):
        """
        Return a list of all correlation function names

        Parameters:

        num_bf: int
            Number of basis functions
        """
        all_cf_names = []
        for cluster in self.clusters:
            all_cf_names += self.get_cf_names(cluster, num_bf)
        return all_cf_names

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, i):
        return self.clusters[i]

    def __array__(self):
        return self.clusters

    def __eq__(self, other):
        print(len(self.clusters), len(other.clusters))
        if len(self.clusters) != len(other.clusters):
            return False

        for c1, c2 in zip(self.clusters, other.clusters):
            if c1 != c2:
                print(c1, c2)
                return False
        return True

    def sort(self):
        self.clusters.sort()

    @property
    def dtype(self):
        return 'ClusterList'

    def tolist(self):
        return self.clusters

    @property
    def unique_indices(self):
        all_indices = set()
        for cluster in self.clusters:
            all_indices.update(flatten(cluster.indices))
        return list(all_indices)

    @property
    def num_symm_groups(self):
        return max(c.group for c in self.clusters)

    @property
    def unique_indices_per_group(self):
        indices_per_group = []
        for i in range(self.num_symm_groups + 1):
            indices_per_group.append(set())
            for c in self.get_by_group(i):
                indices_per_group[i].update(flatten(c.indices))
        return [list(x) for x in indices_per_group]

    def multiplicity_factors(self, num_sites_per_group):
        mult_factors = {}
        norm = {}
        for cluster in self.clusters:
            current_factor = mult_factors.get(cluster.name, 0)
            current_norm = norm.get(cluster.name, 0)
            current_factor += len(cluster.indices)*num_sites_per_group[cluster.group]
            current_norm += num_sites_per_group[cluster.group]

            mult_factors[cluster.name] = current_factor
            norm[cluster.name] = current_norm

        for k in mult_factors.keys():
            mult_factors[k] /= norm[k]
        return mult_factors

    def get_subclusters(self, cluster):
        """Return all all subclusters of the passed cluster in the list."""
        return [c for c in self.clusters if c.is_subcluster(cluster)]
