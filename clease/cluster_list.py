
class ClusterList(object):
    def __init__(self):
        self.clusters = []
        self.parent = {}

    def append(self, cluster):
        self.clusters.append(cluster)

        # Update the lookup
        for sub in cluster.indices:
            key = sorted([cluster.ref_indx] + list(sub))
            v = self.parent.get(tuple(key), [])
            v.append(len(self.clusters))
            self.parent[tuple(key)] = v

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

    def __len__(self):
        return len(self.clusters)
