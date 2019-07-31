from ase.neighborlist import neighbor_list
import numpy as np


class MICDistanceNotUniqueError(Exception):
    pass


class TransMatrixConstructor(object):
    """Class that constructs translation matrices.

    Parameters:

    atoms: Atoms object
        ASE atoms object (assumed to be wrapped and sorted)

    cutoff: float
        Cut-off distance in angstrom
    """

    def __init__(self, atoms, cutoff):
        self.num_atoms = len(atoms)
        self.neighbor = self._construct_neighbor_list(atoms, cutoff)

    def _construct_neighbor_list(self, atoms, cutoff):
        """Construct neighbour list structure."""
        i_first, i_second, d_vec = neighbor_list('ijD', atoms, cutoff)
        # Transfer to more convenienct strucutre
        neighbor = [{"nb_index": [], "dist": []} for _ in range(len(atoms))]

        for i in range(len(i_first)):
            neighbor[i_first[i]]["nb_index"].append(i_second[i])
            neighbor[i_first[i]]["dist"].append(d_vec[i])
        return neighbor

    def _map_one(self, indx, template_indx):
        """Map indices of neighbors to another reference atom."""
        mapped = {template_indx: indx}

        nb_indx = self.neighbor[indx]["nb_index"]
        nb_dist = self.neighbor[indx]["dist"]
        ref_indx = self.neighbor[template_indx]["nb_index"]
        ref_dists = self.neighbor[template_indx]["dist"]

        tol = 1E-6
        for i, d in zip(nb_indx, nb_dist):
            dist_vec = np.array(ref_dists) - np.array(d)
            lengths = np.sum(dist_vec**2, axis=1)
            short_lengths = lengths[lengths < tol]

            if len(short_lengths) > 1:
                raise MICDistanceNotUniqueError(
                    'Multiple atoms has the same distance vector')
            elif len(short_lengths) == 0:
                raise MICDistanceNotUniqueError('No MIC distance vector match')

            corresponding_indx = ref_indx[np.argmin(lengths)]
            mapped[corresponding_indx] = i
        return mapped

    def construct(self, ref_symm_group, symm_group):
        """Construct translation matrix.

        Parameters:

        ref_symm_group: list
            List of reference indices.
            If passed Atoms object has only one basis, ref_symm_group is [0],
            otherwise it hold indices of reference atoms in each basis.
            (e.g., [0, 5, 15] for the Atoms object with 3 basis)

        symm_group: list
            List of the symmetry groups of each Atoms object.
            If passed Atoms object has only one basis, symm_group is
            [0, 0, ..., 0].
            If it has two basis,  this can be [0, 0, 1, 1, 0, 1, ...].
            The index of the reference atom for the basis where the atom
            with index k belongs to is ref_symm_group[symm_group[k]]
        """
        tm = []
        for indx in range(self.num_atoms):
            ref_indx = ref_symm_group[symm_group[indx]]
            mapped = self._map_one(indx, ref_indx)
            tm.append(mapped)
        return tm
