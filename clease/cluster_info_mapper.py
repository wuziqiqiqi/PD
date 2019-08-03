from scipy.spatial import cKDTree as KDTree
from ase.geometry import wrap_positions
from copy import deepcopy
import numpy as np


class AtomsNotContainedInLargeCellError(Exception):
    pass

class ClusterInfoMapper(object):
    """
    Class for accelerating the construction of cluster descriptors when 
    the cluster descriptor already is known for a larger cell

    Parameter
    =========
    atoms: Atoms
        Large atoms cell
    tm_matrix list of dict
        Translation "matrix" for the large cell
    cluster_info: list dict (See settiing)
        Cluster info the for the large cell
    """
    def __init__(self, atoms, tm_matrix, cluster_info):
        self.atoms = atoms
        self.tm_matrix = tm_matrix
        self.cluster_info = cluster_info
        self.tree = KDTree(self.atoms.get_positions())

    def _map_indices(self, small_atoms):
        """
        Map indices from large atoms to small.
        """
        tree = KDTree(small_atoms.get_positions())

        # Positions in large cell
        large_pos = self.atoms.get_positions()
        
        # Wrap these positions into the small cell
        large_pos = wrap_positions(large_pos, small_atoms.get_cell())


        # Now we loop throught the small atoms and finds the closest atoms
        # in the large cell
        dist, index_map = tree.query(large_pos)
        assert np.all(dist < 1E-6)

        if len(list(set(index_map))) != len(small_atoms):
            raise AtomsNotContainedInLargeCellError("All indices not covered")
        return index_map

    def _map_cluster_info(self, index_map):
        """
        Maps the cluster info the old large atoms to the corresponding
        cluster info of the small atoms object
        """
        new_info = deepcopy(self.cluster_info)
        for new_item, old_item in zip(new_info, self.cluster_info):
            for k, v in new_item.items():
                v['ref_indx'] = int(index_map[old_item[k]['ref_indx']])
                o_indx = old_item[k]['indices']
                v['indices'] = [[int(index_map[x]) for x in sub] for sub in o_indx]
        return new_info

    def _map_trans_matrix(self, index_map):
        """
        Map the translation matrix
        """
        unique = list(set(index_map))
        new_tm = []

        for row in unique:
            row_in_large = np.where(index_map == row)[0][0]
            new_tm.append({int(index_map[k]): int(index_map[v])
                           for k, v in self.tm_matrix[row_in_large].items()})
        return new_tm

    def map_info(self, small_atoms):
        """
        Map cluster info from the large host
        """
        index_map = self._map_indices(small_atoms)
        new_info = self._map_cluster_info(index_map)
        new_tm = self._map_trans_matrix(index_map)
        return new_info, new_tm