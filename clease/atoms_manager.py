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