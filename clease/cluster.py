from clease.tools import equivalent_deco


class Cluster(object):
    def __init__(self, size, full_name, indices, order, equiv_sites,
                 trans_symm_group, diameter, ref_indx):
        self.size = size
        self.full_name = full_name
        self.indices = indices
        self.order = order
        self.equiv_sites = equiv_sites
        self.group = trans_symm_group
        self.diameter = diameter
        self.ref_indx = ref_indx

    @property
    def name(self):
        return self.full_name

    def __lt__(self, other):
        """Comparison operator."""
        return (self.size, self.diameter, self.group) < \
            (other.size, other.diameter, other.group)

    def __eq__(self, other):
        return self.full_name == other.full_name

    def equiv_deco(self, deco):
        return equivalent_deco(deco, self.equiv_sites)

    def todict(self):
        """Return a dictionary representation."""
        return {
            'indices': self.indices,
            'size': self.size,
            'order': self.order,
            'symm': self.group,
            'diameter': self.diameter,
            'name': self.full_name,
            'descriptor': self.full_name
        }
