
class ClusterList(object):
    def __init__(self):
        self.clusters = []

        # Maintain a datastructure tracking the parent cluster of all
        # subclusters. Main purpose is to be able to quickly identify
        # which clusters are exactly the same, but belonds to two different
        # translational symmetry groups
        self.parent = {}

    def append(self, cluster):
        self.clusters.append(cluster)

        # Update the lookup
        for sub in cluster.indices:
            key = self._sub_cluster_key(cluster.ref_indx, sub)
            v = self.parent.get(tuple(key), [])
            v.append(len(self.clusters) - 1)
            self.parent[tuple(key)] = v

    def clear(self):
        """
        Clear the content
        """
        self.clusters = []
        self.parent = {}

    def _sub_cluster_key(self, ref_indx, sub):
        key = sorted([ref_indx] + list(sub))
        return tuple(key)

    def names(self):
        return [c.name for c in self.clusters]

    def get_by_size(self, size):
        # Return all clusters with a given size
        return [c for c in self.clusters if c.size == size]

    def get_by_group(self, group):
        """
        Return all clusters in a given symmetry group
        """
        return [c for c in self.clusters if c.group == group]

    def get_equivalent_clusters(self, cluster):
        """
        Return a list with all the equivalent clusters in other
        translational symmetry groups

        Parameters:

        cluster: Cluster
            Instance of cluster that should be matched with its corresponding
            cluster in other translational symmetry groups
        """
        equiv_clusters = set()
        for sub in cluster.indices:
            key = self._sub_cluster_key(cluster.ref_indx, sub)
            parents = self.parents[key]
            equiv_clusters.update(parents)
        equiv = []
        for cluster_id in equiv_clusters:
            equiv.append(self.clusters[cluster_id])
        return equiv

    def __len__(self):
        return len(self.clusters)
