from scipy.spatial import cKDTree as KDTree
from ase.geometry import wrap_positions
from copy import deepcopy
import numpy as np


class ClusterInfoMapper(object):
    def __init__(self, atoms, tm_matrix, cluster_info):
        self.atoms = atoms
        self.tm_matrix = tm_matrix
        self.cluster_info = cluster_info
        self.tree = KDTree(self.atoms.get_positions())

    def map_indices(self, small_atoms):
        """
        Map indices from large atoms to small.
        """
        com = self.atoms.get_center_of_mass()
        com_small = small_atoms.get_center_of_mass()

        # Get the position of the atom that is closest to the center of mass
        dist, ind = self.tree.query(com)
        pos_closest = self.tree.data[ind, :]

        tree = KDTree(small_atoms.get_positions())

        # Positions in large cell
        large_pos = self.atoms.get_positions() - pos_closest
        
        # Wrap these positions into the small cell
        large_pos = wrap_positions(large_pos, small_atoms.get_cell())


        # Now we loop throught the small atoms and finds the closest atoms
        # in the large cell
        dist, index_map = tree.query(large_pos)
        assert np.all(dist < 1E-6)
        return index_map

    def map_cluster_info(self, index_map):
        """
        Maps the cluster info the old large atoms to the corresponding
        cluster info of the small atoms object
        """
        new_trans_mat = {}
        new_info = deepcopy(self.cluster_info)
        for new_item, old_item in zip(new_info, self.cluster_info):
            for k, v in new_item.items():
                v['ref_indx'] = index_map[old_item[k]['ref_indx']]
                o_indx = old_item[k]['indices']
                v['indices'] = [[index_map[x] for x in sub] for sub in o_indx]
        return new_info

    def map_trans_matrix(self, index_map):
        """
        Map the translation matrix
        """
        unique = list(set(index_map))
        new_tm = []

        for row in unique:
            row_in_large = np.where(index_map == row)[0][0]
            new_tm.append({index_map[k]: index_map[v]
                           for k, v in self.tm_matrix[row_in_large].items()})
        return new_tm
