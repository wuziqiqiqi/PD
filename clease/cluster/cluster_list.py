from __future__ import annotations
from itertools import product
import copy
from typing import List, Dict, Any
import logging

from ase import Atoms

from clease.tools import flatten, dec_string, list2str
from clease.jsonio import jsonable
from .cluster_generator import ClusterGenerator
from .cluster import Cluster

logger = logging.getLogger(__name__)
__all__ = ("ClusterList",)


class ClusterDoesNotExistError(Exception):
    """A requested cluster doesn't exist"""


@jsonable("cluster_list")
class ClusterList:
    # pylint: disable=too-many-public-methods
    def __init__(self):
        self._clusters = []
        # Format of the names cache: {num_bf: names}
        self._all_cf_name_cache: Dict[int, List[str]] = {}

    def todict(self) -> Dict[str, Any]:
        return {"clusters": self.clusters}

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> ClusterList:
        cluster_lst = cls()
        clusters = dct["clusters"]
        for cluster in clusters:
            cluster_lst.append(cluster)
        return cluster_lst

    @property
    def clusters(self) -> List[Cluster]:
        return self._clusters

    def append(self, cluster: Cluster) -> None:
        self.clusters.append(cluster)
        self.clear_cache()

    def clear(self) -> None:
        """Clear the content."""
        self.clusters.clear()
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear any cached results."""
        self._all_cf_name_cache.clear()

    @property
    def names(self) -> List[str]:
        """Get all names in the cluster list"""
        return [cluster.name for cluster in self.clusters]

    def get_by_name(self, name) -> List[Cluster]:
        return [cluster for cluster in self.clusters if cluster.name == name]

    def get_by_name_and_group(self, name, group) -> Cluster:
        clusters = self.get_by_name(name)
        for c in clusters:
            if c.group == group:
                return c
        msg = f"There is no cluster named {name} and in group {group}"
        raise ClusterDoesNotExistError(msg)

    def get_by_size(self, size) -> List[Cluster]:
        # Return all clusters with a given size
        return [c for c in self.clusters if c.size == size]

    def max_size(self) -> int:
        return max(c.size for c in self.clusters)

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
    def get_cf_names(cluster: Cluster, num_bf):
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
            cf_names.append(name + "_" + dec_str)
        return sorted(list(set(cf_names)))

    def get_all_cf_names(self, num_bf: int) -> List[str]:
        """
        Return a list of all correlation function names

        :param num_bf: Number of basis functions
        """
        if num_bf in self._all_cf_name_cache:
            # Finding all the CF names can be rather expensive, so
            # we use cached results if possible
            logger.debug("Found a cached result for all cf names for %s num bf", num_bf)
            return self._all_cf_name_cache[num_bf]
        # We havn't cached this name list yet, so we need to build it
        logger.debug("Building all CF names for %s num bf", num_bf)
        all_cf = self._build_all_cf_names(num_bf)
        self._all_cf_name_cache[num_bf] = all_cf
        return all_cf

    def _build_all_cf_names(self, num_bf: int) -> List[str]:
        if not isinstance(num_bf, int):
            raise TypeError(f"Number of basis functions must be integer, got {num_bf}")
        all_cf_names = []
        for cluster in self.clusters:
            if cluster.name == "c0":
                all_cf_names.append("c0")
            else:
                all_cf_names += self.get_cf_names(cluster, num_bf)
        return sorted(set(all_cf_names))

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
                logger.debug("Clusters not equal: %s and %s", c1, c2)
                return False
        return True

    def sort(self):
        """Sort the internal cluster list"""
        self.clusters.sort()
        # We clear this cache, as the ordering has changed, and hence also potentially
        # the ordering of the cached results.
        self.clear_cache()

    def get_sorted_list(self) -> ClusterList:
        """Get a new instance of the ClusterList which is sorted."""
        new_list = copy.deepcopy(self)
        new_list.sort()
        return new_list

    def tolist(self) -> List[Cluster]:
        """Returns a copy of the ClusterList as a regular list."""
        return list(self.clusters)

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
            current_factor += len(cluster.indices) * num_sites_per_group[cluster.group]
            current_norm += num_sites_per_group[cluster.group]

            mult_factors[cluster.name] = current_factor
            norm[cluster.name] = current_norm

        for k in mult_factors:
            mult_factors[k] /= norm[k]
        return mult_factors

    def get_subclusters(self, cluster: Cluster):
        """Return all all subclusters of the passed cluster in the list."""
        return [c for c in self.clusters if c.is_subcluster(cluster)]

    def get_figures(self, generator: ClusterGenerator) -> List[Atoms]:
        """Get the figures (in their ASE Atoms object representation)"""
        figures: List[Atoms] = []
        self.sort()
        # We want to skip c0 and c1 anyways
        used_names = {"c0", "c1"}
        for cluster in self.clusters:
            if cluster.name in used_names:
                continue
            figure = cluster.get_figure(generator.prim)

            figure.info = {"name": cluster.name}
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
            try:
                cluster = self.get_by_name_and_group(c_name, group)
                corr_figure = cluster.corresponding_figure(ref, figure, trans_matrix)
            except (ClusterDoesNotExistError, RuntimeError) as err:
                # If a cutoff is _very_ close to an internal distance in the cell,
                # numerical fluctuations may cause a cluster from one ref site to be included,
                # but not from the other reference site.
                # See !480
                cluster_size = len(figure)
                msg = f"A {cluster_size:d}-body cluster expected to be found was missing. "
                msg += "This *may* indicate a cutoff which is very close to an internal "
                msg += "distance in the cell.\n"
                msg += "If that is the case, try increasing the cutoff diameter for "
                msg += f"{cluster_size:d}-body clusters slightly."
                raise ClusterDoesNotExistError(msg) from err
            corr_fig_key = list2str(corr_figure)
            tot_num += occ_count[cluster.group][corr_fig_key]
        return tot_num

    def make_names_sequential(self):
        """
        Confirm that cluster names are sequential. If clusters are
        removed from the list, that might not be the case.
        """
        for s in range(2, self.max_size() + 1):
            names = list(set(c.name for c in self.get_by_size(s)))
            names.sort()
            prefixes = list(set(n.rpartition("_")[0] for n in names))
            prefixes.sort()
            name_map = {}

            for n in names:
                # Fix distance string
                new_name = copy.deepcopy(n)
                prefix = n.rpartition("_")[0]
                new_dist = f"{prefixes.index(prefix):04d}"
                new_name = "".join([new_name[:4], new_dist, new_name[8:]])
                name_map[n] = new_name

            prefix_map = {}
            # Fix additional ID
            for k in name_map:
                pref = k.rpartition("_")[0]
                prefix_map[pref] = prefix_map.get(pref, []) + [k]

            for k, v in prefix_map.items():
                v.sort()
                uid_map = {}
                for i, x in enumerate(v):
                    new_name = "".join([x.rpartition("_")[0], "_", str(i)])
                    uid_map[x] = new_name

                for k2, v2 in uid_map.items():
                    name_map[k2] = "".join(
                        [name_map[k2].rpartition("_")[0], "_", v2.rpartition("_")[2]]
                    )
            for c in self.get_by_size(s):
                c.name = name_map[c.name]
