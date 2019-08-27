import numpy as np


def list2str(array):
    return "-".join(str(x) for x in array)


class DuplicationCountTracker(object):
    """Tracks duplication counts and normalization factors.

    Parameters:

    cluster_info: list of dicts
        The entire info entry in settings
    """

    def __init__(self, setting):
        self.symm_group = np.zeros(len(setting.atoms), dtype=np.uint8)
        for num, group in enumerate(setting.index_by_trans_symm):
            self.symm_group[group] = num

        self.cluster_info = setting.cluster_info
        self.trans_matrix = setting.trans_matrix
        self.occ_count_all = self._occurence_count_all_symm_groups()
        self._norm_factors = self._get_norm_factors()

    def factor(self, cluster, indices):
        """Get the normalization factor to correct for self interactions.

        Parameters:

        cluster: dict
            Dictionary holding information about the cluster

        indices: list
            Indices of the particular sub cluster
        """
        key = self.index_key(cluster["ref_indx"], indices,
                             cluster["equiv_sites"])
        return self._norm_factors[cluster["symm_group"]][cluster["name"]][key]

    def index_key(self, ref_index, indices, equiv_sites):
        """Return a string representing the key for a given order.

        Parameters:

        ref_index: int
            Reference index

        indices: list
            List representing the indices in a sub-cluster
        """
        return list2str(self._order_equiv_sites(equiv_sites, indices))

    def _get_norm_factors(self):
        """Calculate all normalization factors."""
        norm_factors = []
        for item in self.cluster_info:
            factors = self._get_norm_factors_per_symm_group(item)
            norm_factors.append(factors)
        return norm_factors

    def _get_norm_factors_per_symm_group(self, clusters):
        """Get normalization factors per symmetry group.

        Parameters:

        clusters: dict
            Information dict about all clusters in a symmetry group
        """
        norm_factor = {}
        for name, info in clusters.items():
            occ_count = self.occ_count_all[info["symm_group"]][name]
            norm_factor[name] = self._norm_factor(occ_count, info)
        return norm_factor

    def _occurence_count_all_symm_groups(self):
        occ_count_all = []
        for item in self.cluster_info:
            occ_count = {}
            for name, info in item.items():
                occ_count[name] = self._occurence_count(info)
            occ_count_all.append(occ_count)
        return occ_count_all

    def _occurence_count(self, cluster):
        """Count the number of occurences of each sub-cluster in the cluster

        Parameters:

        cluster: dict
            A dictionary with info about a particular cluster
        """
        occ_count = {}
        for indices in cluster["indices"]:
            key = self.index_key(cluster["ref_indx"], indices,
                                 cluster["equiv_sites"])
            occ_count[key] = occ_count.get(key, 0) + 1
        return occ_count

    def _norm_factor(self, occ_count, cluster):
        norm_count = {}
        for k, v in occ_count.items():
            tot_num = self._total_number_of_occurences(k, cluster["name"])
            num_unique = len(set(k.split("-")))
            norm_count[k] = float(tot_num)/(num_unique*v)
        return norm_count

    def _corresponding_subcluster(self, new_ref_indx, target_cluster, name):
        """Find the corresponding cluster when another index is ref_index."""
        cluster = self.cluster_info[self.symm_group[new_ref_indx]][name]
        for sub in cluster["indices"]:
            indices = [self.trans_matrix[new_ref_indx][indx] for indx in sub]
            indices = self._order_equiv_sites(cluster["equiv_sites"], indices)
            if np.allclose(indices, target_cluster):
                return indices

        raise RuntimeError("There are no matching subcluster. "
                           "This should never happen. This is a bug.")

    def _total_number_of_occurences(self, key, name):
        """Get the total number of occurences."""
        indices = list(map(int, key.split("-")))
        tot_num = 0
        for ref_indx in set(indices):
            corr_cluster = \
                self._corresponding_subcluster(ref_indx, indices, name)
            new_key = list2str(corr_cluster)
            tot_num += \
                self.occ_count_all[self.symm_group[ref_indx]][name][new_key]
        return tot_num

    def show(self):
        """Return a string represenatation."""
        from clease import _logger
        _logger(self._norm_factors)

    def _order_equiv_sites(self, equiv_sites, ordered_indices):
        """After the indices are ordered, adopt a consistent scheme
           within the equivalent sites."""
        for eq_group in equiv_sites:
            equiv_indices = [ordered_indices[i] for i in eq_group]
            equiv_indices.sort()
            for count, i in enumerate(eq_group):
                ordered_indices[i] = equiv_indices[count]
        return ordered_indices
