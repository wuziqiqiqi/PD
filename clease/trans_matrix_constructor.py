from ase.neighborlist import neighbor_list
from clease.tools import ApproxEqualityList


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

        # Re-group by first index
        for i in range(len(i_first)):
            neighbor[i_first[i]]["nb_index"].append(i_second[i].tolist())
            neighbor[i_first[i]]["dist"].append(
                ApproxEqualityList(d_vec[i].tolist()))

        # Sort based on distance
        for i in range(len(neighbor)):
            srt = sorted(list(zip(neighbor[i]["dist"],
                                  neighbor[i]["nb_index"])))
            srt = list(zip(*srt))  # Unzip the list
            neighbor[i]["dist"] = [l.array for l in srt[0]]
            neighbor[i]["nb_index"] = srt[1]
        return neighbor

    def _map_one(self, indx, template_indx):
        """Map indices of neighbors to another reference atom."""
        mapped = {template_indx: indx}

        nb_indx = self.neighbor[indx]["nb_index"]
        ref_indx = self.neighbor[template_indx]["nb_index"]
        mapped = {ref: ind for ref, ind in zip(ref_indx, nb_indx)}
        mapped[template_indx] = indx
        return mapped

    def construct(self, ref_symm_group, symm_group, indices=None):
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

        indices: list of int
            List of indices that the translation matrix should be calculated
            for. If None, all indices will be used.
        """
        tm = []
        if indices is None:
            indices = range(self.num_atoms)

        for indx in indices:
            ref_indx = ref_symm_group[symm_group[indx]]
            mapped = self._map_one(indx, ref_indx)
            tm.append(mapped)
        return tm
