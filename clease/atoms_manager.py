import numpy as np
from math import gcd
from scipy.spatial import cKDTree as KDTree
from ase.geometry import wrap_positions
from ase.build import make_supercell
from clease.tools import wrap_and_sort_by_position
from ase.io import write


class AtomsManager(object):
    """
    Manager class for the Atoms object used in a cluster expansion context.
    This class can return indices of the Atoms object grouped according to
    various schemes.
    """

    def __init__(self, atoms):
        self.atoms = atoms

    def index_by_tag(self):
        """Return atomic indices that are grouped by their tags.

        This method assumes that all atoms are tagged and the tags are in a
        continuous sequence starting from 0.
        """
        tags = set()
        for atom in self.atoms:
            tags.add(atom.tag)

        tags = sorted(list(tags))
        ind_by_tag = [[] for _ in tags]

        for atom in self.atoms:
            ind_by_tag[atom.tag].append(atom.index)
        return ind_by_tag

    def index_by_symbol(self, symbols):
        """Group atomic indices by its atomic symbols.


        Example:

            If symbols = ['Au', 'Cu'], then indices where the indices of Au are
            returned as group 1 (first list in the nested list of indices) and
            the indices of Cu are returned as group 2 (second list in the
            nested list of indices).

            If symbols = ['Au, ['Cu', 'X'], 'Ag'], then indices of Au are
            returned as group 1, the indices of Cu and X are returned as group
            2 and the indices of Ag are returned as group 3.

        Parameters:

        symbols: list
            List with symbols that define a group
        """
        ind_by_symbol = [[] for _ in symbols]
        group_map = {}
        for i, item in enumerate(symbols):
            if isinstance(item, list):
                for x in item:
                    group_map[x] = i
            else:
                group_map[item] = i

        # Loop over indices
        for atom in self.atoms:
            ind_by_symbol[group_map[atom.symbol]].append(atom.index)
        return ind_by_symbol

    def unique_elements(self, ignore=[]):
        """Return a list of symbols of unique elements.

        Parameters:

        ignore: list
            list of symbols to ignore in finding unique elements.
        """
        all_unique = set([a.symbol for a in self.atoms])
        return list(all_unique - set(ignore))

    def single_element_sites(self, allowed_elements):
        """
        Return a list of sites that can only be occupied by a single element
        according to allowed_elements.

        Parameters:

        allowed_elements: list
            List of allowed elements on each site. `allowed_elements` takes the
            same format/style as `basis_elements` in setting (i.e., a nested
            list with each sublist containing a list of elements in a basis).
            It is assumed that all of the first elements in each group is
            present in self.atoms. For example, if `allowed_elements` is
            [['Au', 'Ag' 'X], ['Cu', 'X']] it means that all sites of the
            `self.atoms` must be occupied by either Au or Cu. All of the
            original sites occupied by Au can be occupied by Ag or X in cluster
            expansion, and all of the original sites occupied by Cu can be
            occupied by Cu or X in the cluster expansion.
        """
        single_site_symb = [x[0] for x in allowed_elements if len(x) == 1]
        single_sites = []
        for atom in self.atoms:
            if atom.symbol in single_site_symb:
                single_sites.append(atom.index)
        return single_sites

    def tag_indices_of_corresponding_atom(self, ref_atoms):
        """
        Tag `self.atoms` with the indices of its corresponding atom (equivalent
        position) in `ref_atoms` when the positions are wrapped.


        Parameters:

        ref_atoms: Atoms object
        """
        pos = self.atoms.get_positions()
        wrapped_pos = wrap_positions(pos, ref_atoms.get_cell())
        tree = KDTree(ref_atoms.get_positions())

        dists, indices = tree.query(wrapped_pos)

        if not np.allclose(dists, 0.0):
            raise ValueError("Not all sites has a corresponding atom in the "
                             "passed Atoms object")

        for atom, tag in zip(self.atoms, indices.tolist()):
            atom.tag = tag

    def group_indices_by_trans_symmetry(self, prim_cell):
        """Group indices by translational symmetry.


        Parameters:

        prim_cell: Atoms object
            ASE Atoms object that corresponds to the primitive cell of
            self.atoms
        """
        indices = [a.index for a in prim_cell]
        ref_indx = indices[0]
        # Group all the indices together if its atomic number and position
        # sequences are same
        indx_by_equiv = []
        shifted = wrap_and_sort_by_position(prim_cell.copy())
        an = shifted.get_atomic_numbers()   # atomic number
        pos = shifted.get_positions()       # positions

        temp = [[indices[0]]]
        equiv_group_an = [an]
        equiv_group_pos = [pos]
        for indx in indices[1:]:
            vec = prim_cell.get_distance(indx, ref_indx, vector=True)
            shifted = prim_cell.copy()
            shifted.translate(vec)
            shifted = wrap_and_sort_by_position(shifted)
            an = shifted.get_atomic_numbers()
            pos = shifted.get_positions()

            for equiv_group in range(len(temp)):
                if (an == equiv_group_an[equiv_group]).all() and\
                        np.allclose(pos, equiv_group_pos[equiv_group]):
                    temp[equiv_group].append(indx)
                    break
                else:
                    if equiv_group == len(temp) - 1:
                        temp.append([indx])
                        equiv_group_an.append(an)
                        equiv_group_pos.append(pos)

        for equiv_group in temp:
            indx_by_equiv.append(equiv_group)

        # Now we have found the translational symmetry group of all the atoms
        # in the unit cell, now put all the indices of self.atoms into the
        # matrix based on the tag
        indx_by_equiv_all_atoms = [[] for _ in range(len(indx_by_equiv))]
        symm_group_of_tag = [-1 for _ in range(len(prim_cell))]
        for gr_id, group in enumerate(indx_by_equiv):
            for item in group:
                symm_group_of_tag[item] = gr_id

        for atom in self.atoms:
            symm_gr = symm_group_of_tag[atom.tag]
            indx_by_equiv_all_atoms[symm_gr].append(atom.index)
        return indx_by_equiv_all_atoms

    def corresponding_indices(self, indices, supercell):
        """
        Find the indices in supercell that correspond to the ones in
        self.atoms

        Parameters:

        indices: list of int
            Indices in self.atoms

        supercell: Atoms object
            Supercell object where we want to find the indices
            corresponding to the position in self.atoms
        """
        supercell_indices = []
        sc_pos = supercell.get_positions()
        wrapped_sc_pos = wrap_positions(sc_pos, self.atoms.get_cell())

        dist_to_origin = np.sum(sc_pos**2, axis=1)
        for indx in indices:
            pos = self.atoms[indx].position
            dist = wrapped_sc_pos - pos
            lengths_sq = np.sum(dist**2, axis=1)
            candidates = np.nonzero(lengths_sq < 1E-6)[0].tolist()

            # Pick reference index that is closest the origin of the
            # supercell
            temp_indx = np.argmin(dist_to_origin[candidates])
            supercell_indices.append(candidates[temp_indx])
        return supercell_indices

    def _singlar_cell(self, integer_matrix):
        new_cell = integer_matrix @ self.atoms.cell
        cond = np.linalg.cond(new_cell)
        return cond > 1E10

    def close_to_cubic_supercell(self, zero_cutoff=0.1):
        """
        Create a close to cubic supercell.

        Parameters:

        zero_cutoff: float
            Value below this value will be considered as zero when the
            scaling factor is computed
        """
        cell = self.atoms.get_cell()
        a = np.linalg.det(cell)**(1.0/3.0)
        inv_cell = np.linalg.inv(cell)
        scale = 1.0/inv_cell[np.abs(inv_cell)*a > zero_cutoff]
        scale = np.round(scale).astype(np.int32)
        min_gcd = min([gcd(scale[0], scale[i]) for i in range(len(scale))])

        if min_gcd == 0:
            min_gcd = 1
        scale = np.true_divide(scale, min_gcd)
        scale = min_gcd*np.max(scale)
        integer_matrix = np.round(inv_cell*scale).astype(np.int32)

        counter = 0
        while self._singlar_cell(integer_matrix) and counter < 5:
            row = np.random.randint(low=0, high=3)
            col = np.random.randint(low=0, high=3)
            integer_matrix[row, col] += 1
            counter += 1

        if counter >= 5:
            fname = 'failed_active_template.xyz'
            write(fname, self.atoms)
            raise ValueError("Could not generate a cubic template!"
                             "The activate template is stored in {}"
                             "".format(fname))

        if np.linalg.det(integer_matrix) < 0:
            integer_matrix *= -1

        sc = make_supercell(self.atoms, integer_matrix)
        sc = wrap_and_sort_by_position(sc)

        # We need to tag the atoms
        sc_pos = sc.get_positions()
        sc_pos = wrap_positions(sc_pos, self.atoms.get_cell())

        tree = KDTree(self.atoms.get_positions())
        dists, tags = tree.query(sc_pos)
        assert np.allclose(dists, 0.0)
        for i, tag in enumerate(tags):
            sc[i].tag = tag
        return sc
