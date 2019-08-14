
class ClusterList(object):
    def __init__(self):
        self.clusters = []

    def append(self, cluster):
        self.clusters.append(cluster)

    def clear(self):
        """
        Clear the content
        """
        self.clusters = []

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
        Return equivalent clusters in other symmetry groups
        """
        equiv = []
        for c in self.clusters:
            if c == cluster:
                equiv.append(c)
        return equiv

    def __len__(self):
        return len(self.clusters)
