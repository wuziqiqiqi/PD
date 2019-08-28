from clease.tools import equivalent_deco, nested_array2list
from clease.cluster_fingerprint import ClusterFingerprint



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
            assert self.name >= other.name
        return fp_lt

    def __eq__(self, other):
        """Equality operator."""
        fp_equal = self.fp == other.fp
        if fp_equal:
            assert self.name == other.name
        return fp_equal

    def __ne__(self, other):
        return not self.__eq__(other)

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
                'ref_indx': self.ref_indx,
                'equiv_sites': self.equiv_sites}

    def from_dict(self, data):
        self.indices = data['indices']
        self.size = data['size']
        self.group = data['symm']
        self.diameter = data['diameter']
        self.name = data['name']
        self.fp = data['fingerprint']
        self.ref_indx = data['ref_indx']
        self.equiv_sites = data['equiv_sites']

    @staticmethod
    def load(data):
        cluster = Cluster(None, None, None, None, None, None, None, None)
        fp = ClusterFingerprint.load(data['fingerprint'])
        data['fingerprint'] = fp
        data['indices'] = nested_array2list(data['indices'])
        data['equiv_sites'] = nested_array2list(data['equiv_sites'])
        cluster.from_dict(data)
        return cluster

    def __str__(self):
        str_rep = 'Name: {}\n'.format(self.name)
        str_rep += 'Diameter: {}\n'.format(self.diameter)
        str_rep += 'Size: {}\n'.format(self.size)
        str_rep += 'Ref. indx: {}\n'.format(self.ref_indx)
        str_rep += 'Trans. symm group: {}\n'.format(self.group)
        str_rep += 'Indices: {}\n'.format(self.indices)
        str_rep += 'Equiv. sites: {}\n'.format(self.equiv_sites)
        str_rep += 'Fingerprint: {}\n'.format(self.fp)
        return str_rep
