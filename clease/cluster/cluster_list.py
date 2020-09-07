from itertools import product
from copy import deepcopy
from typing import List, Dict

from clease import _logger
from clease.tools import flatten, dec_string, list2str

__all__ = ('ClusterList',)


class ClusterList(object):

    def __init__(self):
        self.clusters = []
        self.index_by_name = {}

    def append(self, cluster):
        self.clusters.append(cluster)
        idx = self.index_by_name.get(cluster.name, [])
        idx.append(len(self.clusters) - 1)
        self.index_by_name[cluster.name] = idx

    def clear(self):
        """Clear the content."""
        self.clusters = []
        self.index_by_name = {}

    @property
    def names(self):
        return list(self.index_by_name.keys())

    def get_by_name(self, name):
        return [self.clusters[i] for i in self.index_by_name[name]]

    def get_by_name_and_group(self, name, group):
        clusters = self.get_by_name(name)
        for c in clusters:
            if c.group == group:
                return c
        msg = f"There are no cluster named {name} and group {group}"
        raise ValueError(msg)

    def get_by_size(self, size):
        # Return all clusters with a given size
        return [c for c in self.clusters if c.size == size]

    def max_size(self):
        return max([c.size for c in self.clusters])

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

    def get_all_cf_names(self, num_bf: int) -> List[str]:
        """
        Return a list of all correlation function names

        :param num_bf: Number of basis functions
        """
        all_cf_names = []
        for cluster in self.clusters:
            if cluster.name == 'c0':
                all_cf_names.append('c0')
            else:
                all_cf_names += self.get_cf_names(cluster, num_bf)
        return sorted(list(set(all_cf_names)))

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, index):
        return self.clusters[index]

    def __delitem__(self, i):
        del self.clusters[i]

    def __array__(self):
        return self.clusters

    def __eq__(self, other):
        if len(self.clusters) != len(other.clusters):
            return False

        for c1, c2 in zip(self.clusters, other.clusters):
            if c1 != c2:
                _logger(c1, c2)
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

    def multiplicity_factors(self, num_sites_per_group: List[int]) -> Dict[str, float]:
        mult_factors = {}
        norm = {}
        for cluster in self.clusters:
            current_factor = mult_factors.get(cluster.name, 0)
            current_norm = norm.get(cluster.name, 0)
            current_factor += \
                len(cluster.indices)*num_sites_per_group[cluster.group]
            current_norm += num_sites_per_group[cluster.group]

            mult_factors[cluster.name] = current_factor
            norm[cluster.name] = current_norm

        for k in mult_factors.keys():
            mult_factors[k] /= norm[k]
        return mult_factors

    def get_subclusters(self, cluster):
        """Return all all subclusters of the passed cluster in the list."""
        return [c for c in self.clusters if c.is_subcluster(cluster)]

    def get_figures(self, generator):
        figures = []
        self.clusters.sort()
        used_names = set()
        for cluster in self.clusters:
            if (cluster.name == 'c0' or cluster.name == 'c1' or cluster.name in used_names):
                continue
            figure = cluster.get_figure(generator)

            figure.info = {'name': cluster.name}
            used_names.add(cluster.name)
            figures.append(figure)
        return figures

    def get_occurence_counts(self):
        """Count the number of occurences of all figures."""
        occ_counts = []
        for cluster in self.clusters:
            occ_counts.append(cluster.num_fig_occurences)
        return occ_counts

    def num_occ_figure(self, fig_key, c_name, symm_groups, trans_matrix):
        """Determine the number of occurences of a figure across the structure.

        Parameter:

        fig_key: str
            string containing the indices of consituting atomic indices

        c_name: str
            name of the cluster of the figure

        symm_groups: list
            list containing the symmetry group number of the atomic indices

        trans_matrix: list of dicts
            translation matrix
        """
        figure = list(map(int, fig_key.split("-")))
        tot_num = 0
        clusters = self.get_by_name(c_name)
        occ_count = {}
        for cluster in clusters:
            occ_count[cluster.group] = cluster.num_fig_occurences

        for ref in set(figure):
            group = symm_groups[ref]
            cluster = self.get_by_name_and_group(c_name, group)
            corr_figure = \
                cluster.corresponding_figure(ref, figure, trans_matrix)
            corr_fig_key = list2str(corr_figure)
            tot_num += occ_count[cluster.group][corr_fig_key]
        return tot_num

    def make_names_sequential(self):
        """
        Confirm that cluster names are sequential. If clusters are
        removed from the list, that might not be the case.
        """
        for s in range(2, self.max_size() + 1):
            names = list(set([c.name for c in self.get_by_size(s)]))
            names.sort()
            prefixes = list(set([n.rpartition('_')[0] for n in names]))
            prefixes.sort()
            name_map = {}

            for n in names:
                # Fix distance string
                new_name = deepcopy(n)
                prefix = n.rpartition('_')[0]
                new_dist = f"{prefixes.index(prefix):04d}"
                new_name = ''.join([new_name[:4], new_dist, new_name[8:]])
                name_map[n] = new_name

            prefix_map = {}
            # Fix additional ID
            for k in name_map.keys():
                pref = k.rpartition('_')[0]
                prefix_map[pref] = prefix_map.get(pref, []) + [k]

            for k, v in prefix_map.items():
                v.sort()
                uid_map = {}
                for i, x in enumerate(v):
                    new_name = ''.join([x.rpartition('_')[0], '_', str(i)])
                    uid_map[x] = new_name

                for k, v in uid_map.items():
                    name_map[k] = ''.join(
                        [name_map[k].rpartition('_')[0], '_',
                         v.rpartition('_')[2]])
            for c in self.get_by_size(s):
                c.name = name_map[c.name]