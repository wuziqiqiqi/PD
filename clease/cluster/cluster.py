import numpy as np
from ase import Atoms
from .cluster_fingerprint import ClusterFingerprint
from clease.tools import equivalent_deco, nested_array2list, list2str, cname_lt

__all__ = ('Cluster',)


class Cluster(object):

    def __init__(self,
                 name='c0',
                 size=0,
                 diameter=0.0,
                 fingerprint=None,
                 ref_indx=0,
                 indices=(),
                 equiv_sites=(),
                 trans_symm_group=0):
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
            assert cname_lt(self.name, other.name) is True
        else:
            assert cname_lt(self.name, other.name) is False
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
        return {
            'indices': self.indices,
            'size': self.size,
            'symm': self.group,
            'diameter': self.diameter,
            'name': self.name,
            'fingerprint': self.fp.todict(),
            'ref_indx': self.ref_indx,
            'equiv_sites': self.equiv_sites,
            'info': self.info
        }

    def from_dict(self, data):
        self.indices = data['indices']
        self.size = data['size']
        self.group = data['symm']
        self.diameter = data['diameter']
        self.name = data['name']

        if isinstance(data['fingerprint'], ClusterFingerprint):
            self.fp = data['fingerprint']
        elif isinstance(data['fingerprint'], dict):
            self.fp = ClusterFingerprint.load(data['fingerprint'])
        else:
            raise ValueError('Fingerprint has to be either instance of '
                             'ClusterFingerprint or a dictionary')

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

        return any(set(s1).issubset(s2) for s1 in self.indices for s2 in other.indices)

    def get_figure(self, generator):
        if len(self.indices[0][0]) != 4:
            raise RuntimeError("This method requires that the cluster is "
                               "described based on its 4-vector and not index "
                               "in the ASE atoms")

        positions = np.array([generator.cartesian(x) for x in self.indices[0]])
        positions -= np.mean(positions, axis=0)
        sublat_symb = {at.tag: at.symbol for at in generator.prim}
        symbols = [sublat_symb[x[3]] for x in self.indices[0]]
        return Atoms(symbols, positions=positions)

    def __str__(self):
        str_rep = f"Name: {self.name}\n"
        str_rep += f"Diameter: {self.diameter}\n"
        str_rep += f"Size: {self.size}\n"
        str_rep += f"Ref. indx: {self.ref_indx}\n"
        str_rep += f"Trans. symm group: {self.group}\n"
        str_rep += f"Indices: {self.indices}\n"
        str_rep += f"Equiv. sites: {self.equiv_sites}\n"
        str_rep += f"Fingerprint: {self.fp}\n"
        str_rep += f"Information: {self.info}\n"
        return str_rep

    def get_figure_key(self, figure):
        """Return a key representation of the figure."""
        return list2str(self._order_equiv_sites(figure))

    def _order_equiv_sites(self, figure):
        """Sort equivalent sites."""
        figure_cpy = [x for x in figure]
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

        trans_matrix: list of dicts
            translation matrix
        """
        for figure in self.indices:
            translated_figure = [trans_matrix[ref_indx][x] for x in figure]
            translated_figure = self._order_equiv_sites(translated_figure)
            if translated_figure == target_figure:
                return self._order_equiv_sites(figure)

        raise RuntimeError(f"There are no matching figure for ref_indx: "
                           f"{ref_indx} and figure: {target_figure}!")

    def get_all_figure_keys(self):
        return [self.get_figure_key(fig) for fig in self.indices]
