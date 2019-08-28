from clease.tools import equivalent_deco


class Cluster(object):
    def __init__(self,  name, size, diameter, fingerprint, ref_indx, indices,
                 equiv_sites, trans_symm_group):
        self.name = name
        self.size = size
        self.diameter = diameter
        self.fp = fingerprint
        self.ref_indx = ref_indx
        self.indices = indices
        self.equiv_sites = equiv_sites
        self.group = trans_symm_group

    def __lt__(self, other):
        """Comparison operator."""
        fp_lt = self.fp < other.fp
        if fp_lt:
            assert self.name < other.name
        else:
            assert self.name > other.name
        return fp_lt

    def __eq__(self, other):
        """Equality operator."""
        fp_equal = self.fp == other.fp
        if fp_equal:
            assert self.name == other.name
        return fp_equal

    def equiv_deco(self, deco):
        return equivalent_deco(deco, self.equiv_sites)

    def todict(self):
        """Return a dictionary representation."""
        return {'indices': self.indices,
                'size': self.size,
                'symm': self.group,
                'diameter': self.diameter,
                'name': self.name,
                'fingerprint': self.fp,
                'ref_indx': self.ref_indx}

    def from_dict(self, data):
        self.indices = data['indices']
        self.size = data['size']
        self.group = data['symm']
        self.diameter = data['diameter']
        self.name = data['name']
        self.fp = data['fingerprint']
        self.ref_indx = data['ref_indx']
