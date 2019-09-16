import numpy as np
from copy import deepcopy
from clease.tools import equivalent_deco, nested_array2list, list2str
from clease.cluster_fingerprint import ClusterFingerprint


class Cluster(object):
    def __init__(self, name='c0', size=0, diameter=0.0, fingerprint=[],
                 ref_indx=0, indices=[], equiv_sites=[], trans_symm_group=0):
        self.name = name
        self.size = size
        self.diameter = diameter
        self.fp = fingerprint
        self.ref_indx = ref_indx
        self.indices = indices
        self.equiv_sites = equiv_sites
        self.group = trans_symm_group
        self.info = {}

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
                'equiv_sites': self.equiv_sites,
                'info': self.info}

    def from_dict(self, data):
        self.indices = data['indices']
        self.size = data['size']
        self.group = data['symm']
        self.diameter = data['diameter']
        self.name = data['name']
        self.fp = data['fingerprint']
        self.ref_indx = data['ref_indx']
        self.equiv_sites = data['equiv_sites']
        self.info = data['info']

    @staticmethod
    def load(data):
        cluster = Cluster(None, None, None, None, None, None, None, None)
        fp = ClusterFingerprint.load(data['fingerprint'])
        data['fingerprint'] = fp
        data['indices'] = nested_array2list(data['indices'])
        data['equiv_sites'] = nested_array2list(data['equiv_sites'])
        cluster.from_dict(data)
        return cluster

    def is_subcluster(self, other):
        """Check if the passed cluster is a subcluster of the current."""
        if len(self.indices) == 0:
            return True

        if len(self.indices[0]) >= len(other.indices[0]):
            return False

        return any(set(s1).issubset(s2) for s1 in self.indices
                   for s2 in other.indices)

    def get_figure(self, atoms):
        figure = atoms[self.indices[0]]
        figure_center = np.mean(figure.get_positions(), axis=0)
        cell_center = np.mean(figure.get_cell(), axis=0)/2.0
        figure.translate(cell_center - figure_center)
        figure.wrap()
        return figure

    def __str__(self):
        str_rep = 'Name: {}\n'.format(self.name)
        str_rep += 'Diameter: {}\n'.format(self.diameter)
        str_rep += 'Size: {}\n'.format(self.size)
        str_rep += 'Ref. indx: {}\n'.format(self.ref_indx)
        str_rep += 'Trans. symm group: {}\n'.format(self.group)
        str_rep += 'Indices: {}\n'.format(self.indices)
        str_rep += 'Equiv. sites: {}\n'.format(self.equiv_sites)
        str_rep += 'Fingerprint: {}\n'.format(self.fp)
        str_rep += 'Information: {}\n'.format(self.info)
        return str_rep

    def get_figure_key(self, figure):
        """Return a key representation of the figure."""
        return list2str(self._order_equiv_sites(figure))

    def _order_equiv_sites(self, figure):
        """Sort equivalent sites."""
        figure_cpy = deepcopy(figure)
        for eq_group in self.equiv_sites:
            equiv_indices = [figure[i] for i in eq_group]
            equiv_indices.sort()
            for count, i in enumerate(eq_group):
                figure_cpy[i] = equiv_indices[count]
        return figure_cpy

    @property
    def num_fig_occurences(self):
        """Number of currences for each figures."""
        occ_count = {}
        for figure in self.indices:
            key = self.get_figure_key(figure)
            current_num = occ_count.get(key, 0)
            occ_count[key] = current_num + 1
        return occ_count

    def corresponding_figure(self, ref_indx, target_figure, trans_matrix):
        """Find figures that correspond to another reference index.

        Parameters:

        ref_indx: int
            reference index

        taget_figres: list of indices
            list of atomic indices that constitute the original figure before
            translating

        tran_matrix: list of dicts
            translation matrix
        """
        for figure in self.indices:
            translated_figure = [trans_matrix[ref_indx][x] for x in figure]
            translated_figure = self._order_equiv_sites(translated_figure)
            if translated_figure == target_figure:
                return self._order_equiv_sites(figure)

        raise RuntimeError("There are no matching figure!")

    def get_all_figure_keys(self):
        return [self.get_figure_key(fig) for fig in self.indices]
