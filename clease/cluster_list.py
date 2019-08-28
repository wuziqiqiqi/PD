from clease.tools import dec_string
from itertools import product

class ClusterList(object):
    def __init__(self):
        self.clusters = []

    def append(self, cluster):
        self.clusters.append(cluster)

    def clear(self):
        """Clear the content."""
        self.clusters = []

    def names(self):
        return [c.name for c in self.clusters]

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
